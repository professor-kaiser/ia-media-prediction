#ifndef __ML_RF_STRUCTURAL_STACK_FRAME__
#define __ML_RF_STRUCTURAL_STACK_FRAME__

#include <vector>

namespace epsilon::ml::rf::structural
{
	struct StackFrame
	{
		std::vector<int> samples;
		std::vector<float> X;
		std::vector<float> y;
		int depth;
		int cursor;
		int phase;

		float split_gain;
		int split_feature;
		float split_threshold;
		std::vector<int> split_left;
		std::vector<int> split_right;

		int l_root;
		int r_root;
		int index;
	};
}

#endif