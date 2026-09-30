using System;
using Xunit;
using ZeroVector.Core.Indices;
using ZeroVector.Core.Metrics;

namespace ZeroVector.Tests
{
    public class HnswVectorIndexTests
    {
        [Fact]
        public void AddAndSearchTopK_ReturnsExactSelfMatch()
        {
            int dim = 64;
            var hnsw = new HnswVectorIndex(dim, m: 16, efConstruction: 100, efSearch: 50, metric: VectorMetricType.Cosine);
            var rand = new Random(42);

            for (int id = 1; id <= 200; id++)
            {
                var vec = new float[dim];
                for (int d = 0; d < dim; d++)
                {
                    vec[d] = (float)(rand.NextDouble() * 2.0 - 1.0);
                }
                VectorMetrics.NormalizeL2(vec);
                hnsw.Add(id, vec);
            }

            Assert.Equal(200, hnsw.Count);

            var query = new float[dim];
            Assert.True(hnsw.TryGet(100, query));

            var results = hnsw.SearchTopK(query, k: 5);
            Assert.NotEmpty(results);
            Assert.Equal(100, results[0].Id);
            Assert.True(results[0].Score > 0.999f);
        }

        [Fact]
        public void Hnsw_AchievesHighRecallComparedToBruteForce()
        {
            int dim = 32;
            int count = 500;
            var flat = new FlatVectorIndex(dim, count, VectorMetricType.Cosine);
            var hnsw = new HnswVectorIndex(dim, m: 16, efConstruction: 120, efSearch: 60, metric: VectorMetricType.Cosine);
            var rand = new Random(888);

            for (int id = 1; id <= count; id++)
            {
                var vec = new float[dim];
                for (int d = 0; d < dim; d++)
                {
                    vec[d] = (float)(rand.NextDouble() * 2.0 - 1.0);
                }
                VectorMetrics.NormalizeL2(vec);
                flat.Add(id, vec);
                hnsw.Add(id, vec);
            }

            int testQueries = 20;
            int topK = 5;
            int totalHits = 0;

            for (int q = 0; q < testQueries; q++)
            {
                var queryVec = new float[dim];
                for (int d = 0; d < dim; d++)
                {
                    queryVec[d] = (float)(rand.NextDouble() * 2.0 - 1.0);
                }
                VectorMetrics.NormalizeL2(queryVec);

                var exactResults = flat.SearchTopK(queryVec, topK);
                var hnswResults = hnsw.SearchTopK(queryVec, topK);

                var exactSet = new System.Collections.Generic.HashSet<int>();
                for (int i = 0; i < exactResults.Length; i++)
                {
                    exactSet.Add(exactResults[i].Id);
                }

                for (int i = 0; i < hnswResults.Length; i++)
                {
                    if (exactSet.Contains(hnswResults[i].Id))
                    {
                        totalHits++;
                    }
                }
            }

            double recall = (double)totalHits / (testQueries * topK);
            // HNSW with efSearch=60 should achieve > 90% recall
            Assert.True(recall >= 0.85, $"Expected Recall@5 >= 85%, but got {recall:P2}");
        }

        [Fact]
        public void Clear_ResetsCountAndIndex()
        {
            int dim = 16;
            var hnsw = new HnswVectorIndex(dim);
            hnsw.Add(1, new float[16]);
            hnsw.Add(2, new float[16]);
            Assert.Equal(2, hnsw.Count);

            hnsw.Clear();
            Assert.Equal(0, hnsw.Count);
            Assert.Empty(hnsw.SearchTopK(new float[16], 5));
        }
    }
}
