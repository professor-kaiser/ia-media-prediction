#include <set>
#include <stack>
#include <omp.h>
#include "DecisionTree.hpp"

namespace epsilon::ml::rf::structural
{
	DecisionTree::DecisionTree(const int& n)
	{
		count = n;
		thresholds.resize(n);
		features.resize(n);
		lefts.resize(n);
		rights.resize(n);
		labels.resize(n);

		std::fill(lefts.begin(), lefts.end(), -1);
		std::fill(rights.begin(), rights.end(), -1);
	}

	void DecisionTree::set_cursor(int c)
	{
		cursor = c;
	}

	void DecisionTree::add(auto DecisionTree::* v, auto x)
    	requires std::is_arithmetic_v<decltype(x)>
	{
		(this->*v)[cursor] = x;
	}

	int DecisionTree::predict(const std::vector<float>& sample)
	{
		int node = 0;
		const float* __restrict Xd = sample.data();

        while (lefts[node] != -1 || rights[node] != -1)
        {
            float value = Xd[features[node]];
            node = value < thresholds[node] 
            	? lefts[node] 
            	: rights[node];
        }

        return labels[node];
	}

	int DecisionTree::build(
	    const std::vector<float>& X,
	    const std::vector<float>& y,
	    const std::pair<size_t, size_t>& size,
	    const std::pair<int, int>& depth,
	    std::mt19937& rng)
	{
		std::stack<StackFrame> stack;
		StackFrame iframe;
		int max_depth = depth.second;
		size_t index = cursor;
		const size_t SAMPLES_SIZE = size.first;
		const size_t FEATURES_SIZE = size.second; 

		iframe.X = X;
		iframe.depth = depth.first;
		iframe.cursor = cursor;
		iframe.phase = 0;
		iframe.samples.resize(SAMPLES_SIZE);
		std::iota(iframe.samples.begin(), iframe.samples.end(), 0);
		stack.push(iframe);

		while (!stack.empty())
		{
			StackFrame& frame = stack.top();
			const size_t n_samples = frame.samples.size();
			const size_t n_features = FEATURES_SIZE;

			switch(frame.phase)
			{
			case 0:
			{
				size_t m = static_cast<size_t>(std::sqrt(FEATURES_SIZE));

				std::vector<int> labels;
	            labels.reserve(n_samples);

	            for (size_t i = 0; i < n_samples; ++i)
				{
				    const int idx = frame.samples[i];
				    labels.emplace_back(y[idx]);
				}

	            if (frame.depth >= max_depth || 
	            	std::all_of(labels.begin(), labels.end(), [&](int l) { 
	            		return l == labels[0]; 
	            	}))
	            {
	            	this->set_cursor(frame.cursor);
	            	this->add(&DecisionTree::labels, metrics::majority_label(labels));
	            	index = frame.cursor;;
	            	stack.pop();
	            	continue;
	            }

				frame.split_gain = 0;
	            frame.split_feature = -1;

	            float parent_gini = metrics::gini(labels);

			#ifdef __USE_OMP__
	            #pragma omp parallel
	            {
			#endif
	            	std::vector<int> left, right, l_labels, r_labels;
					left.reserve(n_samples);
					right.reserve(n_samples);
					l_labels.reserve(n_samples);
					r_labels.reserve(n_samples);

					// local variable (omp)
					float split_gain = 0.0f;
					int split_feature = 1;
					float split_threshold = 0.0f;
					std::vector<int> split_left;
					std::vector<int> split_right;

				#ifdef __USE_OMP__
	            	#pragma omp parallel for schedule(dynamic)
				#endif
					for (size_t feature = 0; feature < FEATURES_SIZE; feature++)
		            {
		            	const float* __restrict Xf = X.data() + feature;
		            	for (size_t i = 0; i < n_samples - 1; ++i)
						{
							const int idx0 = frame.samples[i];
							const int idx1 = frame.samples[i + 1];

							if (y[idx0] == y[idx1]) continue;

							float threshold = (Xf[idx0 * FEATURES_SIZE] + Xf[idx1 * FEATURES_SIZE]) * 0.5f;

		                    left.clear();
		                    right.clear();
		                    l_labels.clear();
		                    r_labels.clear();

							for (const int& idx : frame.samples)
							{
							    bool go_left = Xf[idx * FEATURES_SIZE] < threshold;
							    (go_left ? left     : right   ).emplace_back(idx);
							    (go_left ? l_labels : r_labels).emplace_back(y[idx]);
							}

		            		if (left.empty() || right.empty()) continue;
	                    
		                    float gain = parent_gini
		                        - (static_cast<float>(left.size()) / n_samples)  * metrics::gini(l_labels)
		                        - (static_cast<float>(right.size()) / n_samples) * metrics::gini(r_labels);
		                    
						#ifdef __USE_OMP__
		                    if (gain > split_gain) 
		                    {
		                        split_gain = gain;
		                        split_feature = feature;
		                        split_threshold = threshold;
		                        split_left = left;
		                        split_right = right;
		                    }
		                #else
		                    if (gain > frame.split_gain) 
		                    {
		                        frame.split_gain = gain;
		                        frame.split_feature = feature;
		                        frame.split_threshold = threshold;
		                        frame.split_left = left;
		                        frame.split_right = right;
		                    }
						#endif
						}
		            }

				#ifdef __USE_OMP__
		            #pragma omp critical
		            {
		            	if (split_gain > frame.split_gain) 
	                    {
	                        frame.split_gain = split_gain;
	                        frame.split_feature = split_feature;
	                        frame.split_threshold = split_threshold;
	                        frame.split_left = std::move(split_left);
	                        frame.split_right = std::move(split_right);
	                    }
		            }
	        	}
				#endif

	            if (frame.split_gain == 0 || frame.cursor >= count) 
                {
	                this->set_cursor(frame.cursor);
	                this->add(&DecisionTree::labels, metrics::majority_label(labels));
	                index = frame.cursor;
	                stack.pop();
	                continue;
	            }

	            frame.index = frame.cursor;
	            this->set_cursor(frame.cursor);
	            this->add(&DecisionTree::features, frame.split_feature);
	            this->add(&DecisionTree::thresholds, frame.split_threshold);
	            
	            frame.l_root = frame.index + 1;
	            frame.phase = 1;
	            
	            StackFrame l_frame;
	            l_frame.samples  = std::move(frame.split_left);
	            l_frame.depth    = frame.depth + 1;
	            l_frame.cursor   = frame.l_root;
	            l_frame.phase    = 0;
	            stack.push(l_frame);
				break;
			}

			case 1:
			{
				frame.index = index;
	            this->set_cursor(frame.cursor);
	            this->add(&DecisionTree::lefts, frame.l_root);
	            
	            frame.r_root = frame.index + 1;
	            frame.phase = 2;
	            
	            StackFrame r_frame;
	            r_frame.samples  = std::move(frame.split_right);
	            r_frame.depth    = frame.depth + 1;
	            r_frame.cursor   = frame.r_root;
	            r_frame.phase    = 0;
	            stack.push(r_frame);
				break;
			}

			default:
			{
				frame.index = index;
	            this->set_cursor(frame.cursor);
	            this->add(&DecisionTree::rights, frame.r_root);
	            
	            index = frame.index;
	            stack.pop();
				break;
			}

			}
		}
	    
	    return index;
	}

	void DecisionTree::print(int node, int depth) const
	{
	    // indentation
	    for (int i = 0; i < depth; ++i)
	        std::cout << "  ";

	    // feuille
	    if (lefts[node] == -1 && rights[node] == -1)
	    {
	        std::cout << "Leaf -> label = " << labels[node] << "\n";
	        return;
	    }

	    // noeud interne
	    std::cout << "Node "
	              << "(feature=" << features[node]
	              << ", threshold=" << thresholds[node]
	              << ")\n";

	    // gauche
	    for (int i = 0; i < depth; ++i)
	        std::cout << "  ";
	    std::cout << "Left:\n";
	    print(lefts[node], depth + 1);

	    // droite
	    for (int i = 0; i < depth; ++i)
	        std::cout << "  ";
	    std::cout << "Right:\n";
	    print(rights[node], depth + 1);
	}

	DecisionTree::~DecisionTree()
	{
		/*delete[] thresholds;
		delete[] features;
		delete[] lefts;
		delete[] rights;
		delete[] labels;*/
	}
}