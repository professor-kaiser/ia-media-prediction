#ifndef __ML_RF_ALGORITHM_METRICS__
#define __ML_RF_ALGORITHM_METRICS__

#include <vector>
#include <random>
#include <unordered_map>

namespace epsilon::ml::rf::algorithm::metrics
{
	const size_t MAX_BINS = 256;
	constexpr size_t PREFETCH_DISTANCE = 16;

	int majority_label(const std::unordered_map<int, int>& freq);
	int majority_label(const std::vector<int>& indices);

	/**
	 * (Classification)
	 * 
	 * gini = 1 - ∑ p_i² | i -> 1...k
	 */
	float gini(const std::vector<int>& indices);
	float gini(const std::unordered_map<int, int>& freq);

	/**
	 * Bootstrap sampling (Bagging)
	 * 
	 * D^[t] = {(x_j, y_j)} for j = 1 to N
	 */
	std::vector<int> bootstrap(int N, std::mt19937& rng);

	void discretize(std::vector<uint8_t>& X_binned, std::vector<float>& bin_edges,
		const std::vector<float>& X, std::pair<size_t, size_t> size);

	void discretize_t(std::vector<uint8_t>& X_binned, std::vector<float>& bin_edges,
		const std::vector<float>& X, std::pair<size_t, size_t> size);

	std::vector<float> transpose(const std::vector<float> &X, std::pair<size_t, size_t> size);
}

#endif