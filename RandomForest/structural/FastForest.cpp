#include "FastForest.hpp"
#include <cstring>
#include <unordered_map>
#include <algorithm>
#include <iostream>

#ifdef __USE_OMP__
	#include <omp.h>
#endif

namespace epsilon::ml::rf::structural
{
	FastForest::FastForest(size_t c) : count(c)
	{
		nodes.resize(c);
	}

	int FastForest::predict(const std::vector<float>& data)
	{
		std::unordered_map<int, int> freq;
		for (size_t i = 0; i < nodes.size(); i++)
		{
			int p = nodes[i]->predict(data);
			++freq[p];
		}

		return std::max_element(freq.begin(), freq.end(),
			[](const auto& a, const auto& b) {
				return a.second < b.second;
			})->first;
	}

	int FastForest::predict(float* data, size_t size)
	{
		std::unordered_map<int, int> freq;
		for (size_t i = 0; i < nodes.size(); i++)
		{
			int p = nodes[i]->predict(data, size);
			++freq[p];
		}

		return std::max_element(freq.begin(), freq.end(),
			[](const auto& a, const auto& b) {
				return a.second < b.second;
			})->first;
	}

	int FastForest::build(
	    const std::vector<float>& X,
	    const std::vector<int>& y,
	    const std::pair<size_t, size_t>& size,
	    const std::pair<int, int>& depth,
	    std::mt19937& rng)
	{
	    int max_depth = depth.second;
	    const size_t FEATURES_SIZE = size.first;
	    const size_t SAMPLES_SIZE = size.second;
	    const size_t TREES_SIZE = (1 << max_depth) - 1;
	    nodes.resize(count);
	    
	#ifdef __USE_OMP__
	    #pragma omp parallel
	    {
	        std::vector<float> X_boot(FEATURES_SIZE * SAMPLES_SIZE);
	        std::vector<int> y_boot(SAMPLES_SIZE);
	        #pragma omp for schedule(dynamic)
	#endif
	        for (size_t c = 0; c < count; c++)
	        {
	    #ifndef __USE_OMP__
	            std::vector<float> X_boot(FEATURES_SIZE * SAMPLES_SIZE);
	            std::vector<int> y_boot(SAMPLES_SIZE);
	    #endif
	            std::shared_ptr<DecisionTree> node = std::make_shared<DecisionTree>(TREES_SIZE);
	            std::vector<int> boot = metrics::bootstrap(SAMPLES_SIZE, rng);

	            constexpr size_t PREFETCH_DISTANCE = 16;
	            #pragma omp simd
	            for (size_t f = 0; f < FEATURES_SIZE; f++)
	            {
	                const float* Xf_src = X.data() + f * SAMPLES_SIZE;
	                float* Xf_boot = X_boot.data() + f * SAMPLES_SIZE;
	                
	                #pragma omp simd
	                for (size_t i = 0; i < SAMPLES_SIZE; i++)
	                {
	                	if (i + PREFETCH_DISTANCE < SAMPLES_SIZE)
	                		__builtin_prefetch(&Xf_src[boot[i + PREFETCH_DISTANCE]], 0, 1);
	                    Xf_boot[i] = Xf_src[boot[i]];
	                }
	            }
	            
	            #pragma omp simd
	            for (size_t i = 0; i < SAMPLES_SIZE; i++)
	            {
	            	if (i + PREFETCH_DISTANCE < SAMPLES_SIZE)
	                	__builtin_prefetch(&y[boot[i + PREFETCH_DISTANCE]], 0, 1);
	                y_boot[i] = y[boot[i]];
	            }
	            
	            node->build(
	                std::move(X_boot),
	                std::move(y_boot),
	                size,
	                depth,
	                rng);
	            nodes[c] = std::move(node);
	        }
	#ifdef __USE_OMP__
	    }
	#endif
	    return 0;
	}

/*	int FastForest::build(
	    const std::vector<float>& X,
	    const std::vector<int>& y,
	    const std::pair<size_t, size_t>& size,
	    const std::pair<int, int>& depth,
	    std::mt19937& rng)
	{
		int max_depth = depth.second;
		const size_t SAMPLES_SIZE = size.second;
		const size_t FEATURES_SIZE = size.first;
		const size_t TREES_SIZE = (1 << max_depth) - 1;

		nodes.resize(count);

	#ifdef __USE_OMP__
		#pragma omp parallel
		{
			std::vector<float> Xt_boot(SAMPLES_SIZE * FEATURES_SIZE);
		    std::vector<int> y_boot(SAMPLES_SIZE);
			#pragma omp for schedule(dynamic)
	#endif
			for (size_t c = 0; c < count; c++)
		    {
		    #ifndef __USE_OMP__
		    	std::vector<float> Xt_boot(SAMPLES_SIZE * FEATURES_SIZE);
		    	std::vector<int> y_boot(SAMPLES_SIZE);
		    #endif

		    	std::shared_ptr<DecisionTree> node = std::make_shared<DecisionTree>(TREES_SIZE);
		    	std::vector<int> boot = metrics::bootstrap(SAMPLES_SIZE, rng);
		    	std::vector<float> Xt = metrics::transpose(X, size);

		        for (size_t i = 0; i < SAMPLES_SIZE; i++)
		        {
		        	const int idx = boot[i];
		            std::memcpy(
		                Xt_boot.data() + i * FEATURES_SIZE,
		                Xt.data() + idx * FEATURES_SIZE,
		                FEATURES_SIZE * sizeof(float)
		            );

		            y_boot[i] = y[idx];
		        }

		        node->build(
		        	std::move(metrics::transpose(Xt_boot, size)), 
		        	std::move(y_boot),
		        	size,
		        	depth,
		        	rng);

		        nodes[c] = std::move(node);
		    }
	#ifdef __USE_OMP__
		}
	#endif

	    return 0;
	}*/
}