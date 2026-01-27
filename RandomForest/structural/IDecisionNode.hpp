#ifndef __ML_RF_STRUCTURAL_IFACE_DECISION_NODE__
#define __ML_RF_STRUCTURAL_IFACE_DECISION_NODE__

#include <vector>
#include <random>

namespace epsilon::ml::rf::structural
{
	class IDecisionNode
	{
	protected:
		inline static std::mt19937& internal_rng()
		{
			static std::mt19937 rng(std::random_device{} ());
			return rng;
		}

	public:
		virtual int build(
		    const std::vector<float>& X,
		    const std::vector<int>& y,
		    const std::pair<size_t, size_t>& size,
		    const std::pair<int, int>& depth,
		    std::mt19937& rng) = 0;
		virtual int predict(const std::vector<float>&) = 0;
		virtual int predict(float* data, size_t size) = 0;
		virtual ~IDecisionNode() = default;	

		template <class Archive>
		void serialize(Archive&)
		{}
	};
}

#endif