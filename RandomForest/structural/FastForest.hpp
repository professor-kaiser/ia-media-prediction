#ifndef __ML_RF_STRUCTURAL_FAST_FOREST__
#define __ML_RF_STRUCTURAL_FAST_FOREST__

#include <vector>
#include <memory>
#include "IDecisionNode.hpp"
#include "DecisionTree.hpp"
#include "../cereal/types/vector.hpp"
#include "../cereal/types/memory.hpp"

namespace metrics = epsilon::ml::rf::algorithm::metrics;

namespace epsilon::ml::rf::structural
{
	class FastForest final : public IDecisionNode
	{
	private:
		std::vector<std::shared_ptr<IDecisionNode>> nodes;
		size_t count;

	public:
		FastForest() = default;
		FastForest(size_t c);

		int predict(const std::vector<float>& data) override;

		int build(
		    const std::vector<float>& X,
		    const std::vector<float>& y,
		    const std::pair<size_t, size_t>& size,
		    const std::pair<int, int>& depth,
		    std::mt19937& rng = internal_rng()) override;

		template <class Archive>
		void serialize(Archive & ar)
		{
			ar(nodes, count);
		}

		~FastForest() = default;
	};
}

#endif