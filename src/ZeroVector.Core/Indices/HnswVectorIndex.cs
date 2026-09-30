using System;
using System.Collections.Generic;
using System.Threading;
using ZeroVector.Core.Metrics;
using ZeroVector.Core.Results;

namespace ZeroVector.Core.Indices
{
    /// <summary>
    /// Pure C# Hierarchical Navigable Small World (HNSW) graph index.
    /// Provides sub-millisecond O(log N) approximate nearest neighbor (ANN) search on high-dimensional vectors.
    /// </summary>
    public sealed class HnswVectorIndex : IVectorIndex
    {
        private readonly int _dimension;
        private readonly int _m;
        private readonly int _m0;
        private readonly int _efConstruction;
        private int _efSearch;
        private readonly double _mL;
        private readonly VectorMetricType _metric;
        private readonly Random _random = new Random(42);
        private readonly ReaderWriterLockSlim _rwLock = new ReaderWriterLockSlim(LockRecursionPolicy.NoRecursion);

        private readonly List<HnswNode> _nodes = new List<HnswNode>();
        private readonly Dictionary<int, int> _idToInternalIndex = new Dictionary<int, int>();
        private int _entryNodeId = -1;
        private int _maxLevel = -1;

        public int Dimension => _dimension;
        public int Count
        {
            get
            {
                _rwLock.EnterReadLock();
                try { return _nodes.Count; }
                finally { _rwLock.ExitReadLock(); }
            }
        }

        public VectorMetricType Metric => _metric;
        public int EfSearch
        {
            get => _efSearch;
            set => _efSearch = Math.Max(1, value);
        }

        public HnswVectorIndex(
            int dimension,
            int m = 16,
            int efConstruction = 100,
            int efSearch = 50,
            VectorMetricType metric = VectorMetricType.Cosine)
        {
            if (dimension <= 0) throw new ArgumentOutOfRangeException(nameof(dimension));
            if (m < 2) throw new ArgumentOutOfRangeException(nameof(m), "M must be at least 2.");

            _dimension = dimension;
            _m = m;
            _m0 = 2 * m;
            _efConstruction = Math.Max(efConstruction, m);
            _efSearch = Math.Max(efSearch, 1);
            _metric = metric;
            _mL = 1.0 / Math.Log(m);
        }

        public void Add(int id, ReadOnlySpan<float> vector)
        {
            if (vector.Length != _dimension)
                throw new ArgumentException($"Vector dimension {vector.Length} does not match index dimension {_dimension}.");

            _rwLock.EnterWriteLock();
            try
            {
                if (_idToInternalIndex.ContainsKey(id))
                {
                    // Existing node: ignore duplicate ID or could update
                    return;
                }

                float[] vecArray = vector.ToArray();
                int level = SampleRandomLevel();
                int internalId = _nodes.Count;
                var newNode = new HnswNode(id, internalId, vecArray, level);
                _nodes.Add(newNode);
                _idToInternalIndex[id] = internalId;

                if (_entryNodeId == -1)
                {
                    _entryNodeId = internalId;
                    _maxLevel = level;
                    return;
                }

                int currObj = _entryNodeId;
                float dist = ComputeDistance(vector, _nodes[currObj].Vector);

                // 1. Search from top level down to level + 1 (greedy search)
                for (int l = _maxLevel; l > level; l--)
                {
                    bool changed = true;
                    while (changed)
                    {
                        changed = false;
                        var neighbors = _nodes[currObj].GetNeighbors(l);
                        for (int i = 0; i < neighbors.Count; i++)
                        {
                            int neighborId = neighbors[i];
                            float d = ComputeDistance(vector, _nodes[neighborId].Vector);
                            if (d < dist)
                            {
                                dist = d;
                                currObj = neighborId;
                                changed = true;
                            }
                        }
                    }
                }

                // 2. Search from min(maxLevel, level) down to 0 and establish connections
                for (int l = Math.Min(_maxLevel, level); l >= 0; l--)
                {
                    var candidates = SearchLayer(vector, currObj, _efConstruction, l);
                    int maxM = (l == 0) ? _m0 : _m;

                    // Select neighbors and connect mutually
                    var selected = SelectNeighborsSimple(candidates, maxM);
                    foreach (var neighbor in selected)
                    {
                        newNode.AddNeighbor(l, neighbor.InternalId);
                        _nodes[neighbor.InternalId].AddNeighbor(l, internalId);

                        // Shrink neighbor connections if exceeding maxM
                        ShrinkConnections(_nodes[neighbor.InternalId], l, maxM);
                    }

                    if (candidates.Count > 0)
                    {
                        currObj = candidates[0].InternalId;
                    }
                }

                if (level > _maxLevel)
                {
                    _maxLevel = level;
                    _entryNodeId = internalId;
                }
            }
            finally
            {
                _rwLock.ExitWriteLock();
            }
        }

        public bool TryGet(int id, Span<float> destination)
        {
            if (destination.Length < _dimension) return false;

            _rwLock.EnterReadLock();
            try
            {
                if (!_idToInternalIndex.TryGetValue(id, out int internalId))
                    return false;

                new ReadOnlySpan<float>(_nodes[internalId].Vector).CopyTo(destination);
                return true;
            }
            finally
            {
                _rwLock.ExitReadLock();
            }
        }

        public VectorSearchResult[] SearchTopK(ReadOnlySpan<float> query, int k)
        {
            return SearchTopK(query, k, _metric);
        }

        public VectorSearchResult[] SearchTopK(ReadOnlySpan<float> query, int k, VectorMetricType metric)
        {
            if (query.Length != _dimension)
                throw new ArgumentException($"Query vector dimension {query.Length} must match index dimension {_dimension}.");

            if (k <= 0) return Array.Empty<VectorSearchResult>();

            _rwLock.EnterReadLock();
            try
            {
                if (_nodes.Count == 0 || _entryNodeId == -1)
                    return Array.Empty<VectorSearchResult>();

                int currObj = _entryNodeId;
                float dist = ComputeDistance(query, _nodes[currObj].Vector);

                // 1. Greedy routing down to level 1
                for (int l = _maxLevel; l > 0; l--)
                {
                    bool changed = true;
                    while (changed)
                    {
                        changed = false;
                        var neighbors = _nodes[currObj].GetNeighbors(l);
                        for (int i = 0; i < neighbors.Count; i++)
                        {
                            int neighborId = neighbors[i];
                            float d = ComputeDistance(query, _nodes[neighborId].Vector);
                            if (d < dist)
                            {
                                dist = d;
                                currObj = neighborId;
                                changed = true;
                            }
                        }
                    }
                }

                // 2. Beam search on layer 0 with size max(k, efSearch)
                int ef = Math.Max(k, _efSearch);
                var candidates = SearchLayer(query, currObj, ef, 0);

                int actualK = Math.Min(k, candidates.Count);
                var results = new VectorSearchResult[actualK];

                for (int i = 0; i < actualK; i++)
                {
                    var cand = candidates[i];
                    int extId = _nodes[cand.InternalId].ExternalId;
                    float score = ScoreFromDistance(cand.Distance, metric, query, _nodes[cand.InternalId].Vector);
                    results[i] = new VectorSearchResult(extId, score);
                }

                return results;
            }
            finally
            {
                _rwLock.ExitReadLock();
            }
        }

        public void Clear()
        {
            _rwLock.EnterWriteLock();
            try
            {
                _nodes.Clear();
                _idToInternalIndex.Clear();
                _entryNodeId = -1;
                _maxLevel = -1;
            }
            finally
            {
                _rwLock.ExitWriteLock();
            }
        }

        #region Internal HNSW Helpers

        private List<Candidate> SearchLayer(ReadOnlySpan<float> query, int entryPoint, int ef, int level)
        {
            var visited = new HashSet<int> { entryPoint };
            var candidateQueue = new ZeroPriorityQueue<Candidate, float>(); // Min-heap for exploration
            var results = new List<Candidate>(ef);

            float entryDist = ComputeDistance(query, _nodes[entryPoint].Vector);
            var entryCand = new Candidate(entryPoint, entryDist);

            candidateQueue.Enqueue(entryCand, entryDist);
            results.Add(entryCand);

            while (candidateQueue.Count > 0)
            {
                var curr = candidateQueue.Dequeue();
                float furthestResultDist = results[results.Count - 1].Distance;

                if (curr.Distance > furthestResultDist && results.Count >= ef)
                {
                    break;
                }

                var neighbors = _nodes[curr.InternalId].GetNeighbors(level);
                for (int i = 0; i < neighbors.Count; i++)
                {
                    int neighborId = neighbors[i];
                    if (visited.Add(neighborId))
                    {
                        float d = ComputeDistance(query, _nodes[neighborId].Vector);
                        furthestResultDist = results[results.Count - 1].Distance;

                        if (d < furthestResultDist || results.Count < ef)
                        {
                            var cand = new Candidate(neighborId, d);
                            candidateQueue.Enqueue(cand, d);
                            InsertSorted(results, cand, ef);
                        }
                    }
                }
            }

            return results;
        }

        private static void InsertSorted(List<Candidate> list, Candidate item, int maxCapacity)
        {
            int idx = list.BinarySearch(item, CandidateComparer.Instance);
            if (idx < 0) idx = ~idx;

            if (idx < maxCapacity)
            {
                list.Insert(idx, item);
                if (list.Count > maxCapacity)
                {
                    list.RemoveAt(list.Count - 1);
                }
            }
        }

        private List<Candidate> SelectNeighborsSimple(List<Candidate> candidates, int m)
        {
            int count = Math.Min(candidates.Count, m);
            var res = new List<Candidate>(count);
            for (int i = 0; i < count; i++)
            {
                res.Add(candidates[i]);
            }
            return res;
        }

        private void ShrinkConnections(HnswNode node, int level, int maxM)
        {
            var neighbors = node.GetNeighbors(level);
            if (neighbors.Count <= maxM) return;

            // Sort neighbors by distance to node and keep closest maxM
            var scoredNeighbors = new List<Candidate>(neighbors.Count);
            for (int i = 0; i < neighbors.Count; i++)
            {
                int nId = neighbors[i];
                float d = ComputeDistance(node.Vector, _nodes[nId].Vector);
                scoredNeighbors.Add(new Candidate(nId, d));
            }

            scoredNeighbors.Sort(CandidateComparer.Instance);
            neighbors.Clear();
            for (int i = 0; i < maxM; i++)
            {
                neighbors.Add(scoredNeighbors[i].InternalId);
            }
        }

        private float ComputeDistance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            return _metric switch
            {
                VectorMetricType.Cosine => 1.0f - VectorMetrics.CosineSimilarity(a, b),
                VectorMetricType.DotProduct => -VectorMetrics.DotProduct(a, b),
                VectorMetricType.Euclidean => VectorMetrics.EuclideanDistance(a, b),
                VectorMetricType.EuclideanSquared => VectorMetrics.EuclideanDistanceSquared(a, b),
                VectorMetricType.Manhattan => VectorMetrics.ManhattanDistance(a, b),
                _ => 1.0f - VectorMetrics.CosineSimilarity(a, b)
            };
        }

        private static float ScoreFromDistance(float dist, VectorMetricType metric, ReadOnlySpan<float> q, ReadOnlySpan<float> v)
        {
            return metric switch
            {
                VectorMetricType.Cosine => VectorMetrics.CosineSimilarity(q, v),
                VectorMetricType.DotProduct => VectorMetrics.DotProduct(q, v),
                VectorMetricType.Euclidean => dist,
                VectorMetricType.EuclideanSquared => dist,
                VectorMetricType.Manhattan => dist,
                _ => 1.0f - dist
            };
        }

        private int SampleRandomLevel()
        {
            double r = _random.NextDouble();
            if (r == 0) r = 0.0000001;
            int lvl = (int)(-Math.Log(r) * _mL);
            return Math.Min(lvl, 16);
        }

        #endregion

        #region Nested Types

        private sealed class HnswNode
        {
            public int ExternalId { get; }
            public int InternalId { get; }
            public float[] Vector { get; }
            public int Level { get; }
            private readonly List<int>[] _neighbors;

            public HnswNode(int externalId, int internalId, float[] vector, int level)
            {
                ExternalId = externalId;
                InternalId = internalId;
                Vector = vector;
                Level = level;
                _neighbors = new List<int>[level + 1];
                for (int i = 0; i <= level; i++)
                {
                    _neighbors[i] = new List<int>();
                }
            }

            public List<int> GetNeighbors(int level) => _neighbors[level];

            public void AddNeighbor(int level, int neighborInternalId)
            {
                var list = _neighbors[level];
                if (!list.Contains(neighborInternalId))
                {
                    list.Add(neighborInternalId);
                }
            }
        }

        private readonly struct Candidate
        {
            public int InternalId { get; }
            public float Distance { get; }

            public Candidate(int internalId, float distance)
            {
                InternalId = internalId;
                Distance = distance;
            }
        }

        private sealed class CandidateComparer : IComparer<Candidate>
        {
            public static readonly CandidateComparer Instance = new CandidateComparer();
            public int Compare(Candidate x, Candidate y) => x.Distance.CompareTo(y.Distance);
        }

        #endregion
    }
}
