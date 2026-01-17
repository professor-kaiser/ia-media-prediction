#ifndef __ML_RF_ALGORITHM_METRICS__
#define __ML_RF_ALGORITHM_METRICS__

#include <vector>
#include <random>

namespace epsilon::ml::rf::algorithm::metrics
{
	int majority_label(const std::vector<int>& indices);

	/**
	 * (Classification)
	 * 
	 * gini = 1 - ∑ p_i² | i -> 1...k
	 */
	float gini(const std::vector<int>& indices);

	/**
	 * Bootstrap sampling (Bagging)
	 * 
	 * D^[t] = {(x_j, y_j)} for j = 1 to N
	 */
	std::vector<int> bootstrap(int N, std::mt19937& rng);
}

#endif