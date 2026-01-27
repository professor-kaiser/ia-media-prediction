#include "metrics.hpp"
#include <unordered_map>
#include <map>
#include <cstring>
#include <algorithm>

namespace epsilon::ml::rf::algorithm::metrics
{
	int majority_label(const std::unordered_map<int, int>& freq)
	{
		return std::max_element(freq.begin(), freq.end(),
			[](const auto& a, const auto& b) {
				return a.second < b.second;
			})->first;
	}

	int majority_label(const std::vector<int>& indices)
	{
		std::unordered_map<int, int> freq;

	#ifdef __USE_OMP__
		#pragma omp parallel
		{
			#pragma omp for schedule(dynamic)
			for (size_t i = 0; i < indices.size(); i++)
			{
				#pragma omp critical
				++freq[indices[i]];
			}
		}
	#else
		for (const int& idx : indices)
		{
			++freq[idx];
		}
	#endif

		return std::max_element(freq.begin(), freq.end(),
			[](const auto& a, const auto& b) {
				return a.second < b.second;
			})->first;
	}

	float gini(const std::vector<int>& indices)
	{
		int n = indices.size();
		std::unordered_map<int, int> freq;
		
	#ifdef __USE_OMP__
		#pragma omp parallel
		{
			#pragma omp for schedule(dynamic)
			for (size_t i = 0; i < indices.size(); i++)
			{
				#pragma omp critical
				++freq[indices[i]];
			}
		}
	#else
		for (const int& idx : indices)
		{
			++freq[idx];
		}
	#endif

		float gini = 1.0f;
		for (const auto& [group, count] : freq)
		{
			float p = static_cast<float>(count) / n;
			gini -= p*p;
		}

		return gini;
	}

	float gini(const std::unordered_map<int, int>& freq)
	{
		int n = 0;
		int sum_sq = 0;

		for (const auto& [group, count] : freq)
		{
			n += count;
			sum_sq += count * count;
		}

		return n == 0 
			? 0.0f
			: 1 - static_cast<float>(sum_sq) / (n * n);
	}
	
	std::vector<int> bootstrap(int N, std::mt19937& rng)
	{
		std::uniform_int_distribution<int> dist(0, N-1);
		std::vector<int> indices(N);
		//indices.reserve(N);

	#ifdef __USE_OMP__
		#pragma omp parallel for schedule(dynamic)
	#endif
		for (int i = 0; i < N; i++)
		{
			indices[i] = dist(rng);
			//indices.emplace_back(dist(rng));
		}

		return indices;
	}

	void discretize(std::vector<uint8_t>& X_binned, std::vector<float>& bin_edges,
		const std::vector<float>& X, std::pair<size_t, size_t> size)
	{
		const auto& [SAMPLES_SIZE, FEATURES_SIZE] = size;

		X_binned.resize(SAMPLES_SIZE * FEATURES_SIZE);
		bin_edges.resize((MAX_BINS + 1) * FEATURES_SIZE);
		#pragma omp parallel for schedule(dynamic)
	    for (size_t feat = 0; feat < FEATURES_SIZE; ++feat)
	    {
	        const float* Xf = X.data() + feat;
	        uint8_t* Xf_binned = X_binned.data() + feat;
	        std::vector<float> data_feature(SAMPLES_SIZE);
	        
			#pragma omp simd
	        for (size_t i = 0; i < SAMPLES_SIZE; ++i)
	        {
	            data_feature[i] = Xf[i * FEATURES_SIZE];
	        }
	        
	        std::sort(data_feature.begin(), data_feature.end());
	        data_feature.erase(
	        	std::unique(data_feature.begin(), data_feature.end()), 
	        	data_feature.end());
	        
	        size_t n_bins = std::min(data_feature.size(), MAX_BINS);
	        
	        float* edges = bin_edges.data() + feat * (MAX_BINS + 1);

	        #pragma omp simd
	        for (size_t b = 0; b < n_bins; ++b)
	        {
	            edges[b] = data_feature[b * data_feature.size() / n_bins];
	        }
	        edges[n_bins] = data_feature.back() + 1e-5f;

			#pragma omp simd
	        for (size_t i = 0; i < SAMPLES_SIZE; ++i)
	        {
	            float val = Xf[i * FEATURES_SIZE];
	            auto it = std::upper_bound(edges, edges + n_bins + 1, val);
	            Xf_binned[i * FEATURES_SIZE] = static_cast<uint8_t>(std::distance(edges, it) - 1);
	        }
	    }
	}

	void discretize_t(std::vector<uint8_t>& X_binned, std::vector<float>& bin_edges,
		const std::vector<float>& X, std::pair<size_t, size_t> size)
	{
		const auto& [FEATURES_SIZE, SAMPLES_SIZE] = size;
		X_binned.resize(SAMPLES_SIZE * FEATURES_SIZE);
		bin_edges.resize((MAX_BINS + 1) * FEATURES_SIZE);

		#pragma omp parallel for schedule(dynamic)
	    for (size_t feat = 0; feat < FEATURES_SIZE; ++feat)
	    {
	        const float* Xf = X.data() + feat * SAMPLES_SIZE;
	        uint8_t* Xf_binned = X_binned.data() + feat * SAMPLES_SIZE;
	        std::vector<float> data_feature(SAMPLES_SIZE, 0.0f);

	        //std::memcpy(data_feature.data(), Xf, SAMPLES_SIZE * sizeof(float));

	        data_feature.assign(Xf, Xf + SAMPLES_SIZE);
	        std::sort(data_feature.begin(), data_feature.end());
	        data_feature.erase(
	        	std::unique(data_feature.begin(), data_feature.end()), 
	        	data_feature.end());
	        
	        size_t n_bins = std::min(data_feature.size(), MAX_BINS);
	        
	        float* edges = bin_edges.data() + feat * (MAX_BINS + 1);

	        #pragma omp simd
	        for (size_t b = 0; b < n_bins; ++b)
	        {
	            edges[b] = data_feature[b * data_feature.size() / n_bins];
	        }
	        edges[n_bins] = data_feature.back() + 1e-5f;

			#pragma omp simd
	        for (size_t i = 0; i < SAMPLES_SIZE; ++i)
	        {
	            float val = Xf[i];
	            auto it = std::upper_bound(edges, edges + n_bins + 1, val);
	            Xf_binned[i] = static_cast<uint8_t>(std::distance(edges, it) - 1);
	        }
	    }
	}

	std::vector<float> transpose(const std::vector<float> &X, std::pair<size_t, size_t> size)
	{
		const auto& [ROWS, COLUMNS] = size;
		std::vector<float> Xt(ROWS * COLUMNS);

		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < ROWS; i++)
		{
			for (size_t j = 0; j < COLUMNS; j++)
			{
				Xt[j * ROWS + i] = X[i * COLUMNS + j];
			}
		}

		return Xt;
	}
}