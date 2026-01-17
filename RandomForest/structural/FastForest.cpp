#include "FastForest.hpp"
#include <cstring>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <omp.h>

namespace epsilon::ml::rf::structural
{
	FastForest::FastForest(size_t c) : count(c)
	{
		nodes.reserve(c);
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

	int FastForest::build(
	    const std::vector<float>& X,
	    const std::vector<float>& y,
	    const std::pair<size_t, size_t>& size,
	    const std::pair<int, int>& depth,
	    std::mt19937& rng)
	{
		int max_depth = depth.second;
		const size_t SAMPLES_SIZE = size.first;
		const size_t FEATURES_SIZE = size.second;

	#ifdef __USE_OMP__
		#pragma omp parallel for schedule(dynamic)
	#endif
		for (size_t c = 0; c < count; c++)
	    {
	    	std::shared_ptr<IDecisionNode> node = std::make_shared<DecisionTree>((1 << max_depth) - 1);
	        std::vector<int> boot = metrics::bootstrap(SAMPLES_SIZE, rng);
	        const size_t SAMPLES_BOOT_SIZE = boot.size();
	        
	        std::vector<std::pair<float, int>> sorted_boot;
	        sorted_boot.reserve(SAMPLES_BOOT_SIZE);
	        for (size_t i = 0; i < SAMPLES_BOOT_SIZE; i++)
	        {
	            sorted_boot.emplace_back(y[boot[i]], boot[i]);
	        }

	        std::sort(sorted_boot.begin(), sorted_boot.end());
	        
	        std::vector<float> X_boot(SAMPLES_BOOT_SIZE * FEATURES_SIZE);
	        std::vector<float> y_boot(SAMPLES_BOOT_SIZE);
	        for (size_t i = 0; i < SAMPLES_BOOT_SIZE; i++)
	        {
	        	auto [label, idx] = sorted_boot[i];

	            std::memcpy(
	                X_boot.data() + i * FEATURES_SIZE,
	                X.data() + idx * FEATURES_SIZE,
	                FEATURES_SIZE * sizeof(float));

	            y_boot[i] = label;
	        }

	        node->build(
	        	std::move(X_boot), 
	        	std::move(y_boot), 
	        	std::make_pair(SAMPLES_BOOT_SIZE, FEATURES_SIZE), 
	        	depth, 
	        	rng);

	        nodes.push_back(node);
	    }

	    return 0;
	}
}