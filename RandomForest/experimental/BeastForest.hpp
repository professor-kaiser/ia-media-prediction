#ifndef __ML_RF_EXPERIMENTAL_BEAST_FOREST__
#define __ML_RF_EXPERIMENTAL_BEAST_FOREST__

#include <vector>
#include <random>
#include <algorithm>

namespace epsilon::ml::rf::experimental
{
	struct BeastForest
	{
		std::vector<float> thresholds;
		std::vector<int> features;
		std::vector<int> lefts;
		std::vector<int> rights;
		std::vector<int> labels;
		size_t offset  = 0;
		size_t n_trees = 0;
		int cursor = 0;
		int n_bins = 256;
    int n_classes = 0;

		BeastForest(const size_t& n_trees, const size_t& offset);

		int predict(const std::vector<float>& data);
		int predict_tree(const std::vector<float>& data, const int& page);

		void compute_n_classes(const std::vector<int>& y) {
	        if (y.empty()) {
	            n_classes = 2;
	            return;
	        }
	        
	        int max_class = *std::max_element(y.begin(), y.end());
	        n_classes = max_class + 1;
	    }

		int build(
			const std::vector<float>& X,
		    const std::vector<int>& y,
		    const std::pair<size_t, size_t>& size,
		    const std::pair<int, int>& depth);

		int build_tree(
		    const std::vector<float>& X,
		    const std::vector<int>& y,
		    const std::pair<size_t, size_t>& size,
		    const std::pair<int, int>& depth,
		    const int& page,
		    std::mt19937& rng);
	};
}

#endif