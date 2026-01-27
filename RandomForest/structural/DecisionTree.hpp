#ifndef __ML_RF_STRUCTURAL_DECISION_NODE__
#define __ML_RF_STRUCTURAL_DECISION_NODE__

#include <random>
#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>
#include <memory>
#include <concepts>
#include <iostream>
#include <unordered_set>
#include "../cereal/types/vector.hpp"
#include "StackFrame.hpp"
#include "IDecisionNode.hpp"
#include "../algorithm/metrics.hpp"

namespace metrics = epsilon::ml::rf::algorithm::metrics;

namespace epsilon::ml::rf::structural
{
	class DecisionTree final : public IDecisionNode
	{
	public:
		DecisionTree() = default;
		DecisionTree(const int& n);
		DecisionTree(const DecisionTree&) = delete;
    	DecisionTree& operator=(const DecisionTree&) = delete;

    	void set_cursor(int c);
    	void resize(int c);

    	void add(auto DecisionTree::* v, auto x)
    		requires std::is_arithmetic_v<decltype(x)>;

		int predict(const std::vector<float>& data) override;
		int predict(float* data, size_t size) override;

		int build(
		    const std::vector<float>& X,
		    const std::vector<int>& y,
		    const std::pair<size_t, size_t>& size,
		    const std::pair<int, int>& depth,
		    std::mt19937& rng = internal_rng()) override;

		void print(int node = 0, int depth = 0) const;

		template <class Archive>
		void serialize(Archive & ar)
		{
			ar(
				thresholds,
				features,
				lefts,
				rights,
				labels,
				count,
				cursor);
		}

		~DecisionTree();

	private:
		std::vector<float> thresholds;
		std::vector<int> features;
		std::vector<int> lefts;
		std::vector<int> rights;
		std::vector<int> labels;
		std::vector<int> boot;
		int count = 0;
		int cursor = 0;
	};
}

#endif