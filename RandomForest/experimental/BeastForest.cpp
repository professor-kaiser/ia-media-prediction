#include "BeastForest.hpp"
#include "../algorithm/metrics.hpp"
#include "../structural/StackFrame.hpp"
#include <algorithm>
#include <cstring>
#include <random>
#include <stack>

#ifdef __USE_OMP__
	#include <omp.h>
#endif

namespace metrics = epsilon::ml::rf::algorithm::metrics;
using epsilon::ml::rf::structural::StackFrame;

namespace epsilon::ml::rf::experimental
{
	BeastForest::BeastForest(const size_t& n_trees, const size_t& offset)
	{
		this->n_trees = n_trees;
		this->offset = offset;

		thresholds.resize(n_trees * offset);
		features  .resize(n_trees * offset);
		lefts     .resize(n_trees * offset);
		rights    .resize(n_trees * offset);
		labels    .resize(n_trees * offset);

		std::fill(lefts.begin(), lefts.end(), -1);
		std::fill(rights.begin(), rights.end(), -1);
	}
	
	int BeastForest::predict(const std::vector<float>& data)
	{
		std::unordered_map<int, int> freq;
		for (size_t page = 0; page < n_trees; page++)
		{
			int p = predict_tree(data, page);
			++freq[p];
		}

		return std::max_element(freq.begin(), freq.end(),
			[](const auto& a, const auto& b) {
				return a.second < b.second;
			})->first;
	}

	int BeastForest::predict_tree(const std::vector<float>& sample, const int& page)
	{
		int node = page * offset;
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

	int BeastForest::build(
		const std::vector<float>& X,
	    const std::vector<int>& y,
	    const std::pair<size_t, size_t>& size,
	    const std::pair<int, int>& depth)
	{
		int max_depth = depth.second;
		const size_t SAMPLES_SIZE = size.first;
		const size_t FEATURES_SIZE = size.second;
		const size_t TREES_SIZE = offset;
		std::mt19937 rng(std::random_device{} ());

	#ifdef __USE_OMP__
		#pragma omp parallel
		{
			std::vector<float> X_boot(SAMPLES_SIZE * FEATURES_SIZE);
		    std::vector<int> y_boot(SAMPLES_SIZE);

			#pragma omp for schedule(dynamic)
	#endif
			for (size_t page = 0; page < n_trees; page++)
		    {
		    #ifndef __USE_OMP__
		    	std::vector<float> X_boot(SAMPLES_SIZE * FEATURES_SIZE);
		    	std::vector<int> y_boot(SAMPLES_SIZE);
		    #endif

		    	std::vector<int> boot = metrics::bootstrap(SAMPLES_SIZE, rng);
		        for (size_t i = 0; i < SAMPLES_SIZE; i++)
		        {
		        	const int idx = boot[i];
		            std::memcpy(
		                X_boot.data() + i * FEATURES_SIZE,
		                X.data() + idx * FEATURES_SIZE,
		                FEATURES_SIZE * sizeof(float)
		            );

		            y_boot[i] = y[idx];
		        }

		        build_tree(
		        	std::move(X_boot), 
		        	std::move(y_boot),
		        	std::make_pair(SAMPLES_SIZE, FEATURES_SIZE),
		        	depth,
		        	page,
		        	rng);
		    }
	#ifdef __USE_OMP__
		}
	#endif

	    return 0;
	}

	int BeastForest::build_tree(
	    const std::vector<float>& X,
	    const std::vector<int>& y,
	    const std::pair<size_t, size_t>& size,
	    const std::pair<int, int>& depth,
	    const int& page,
	    std::mt19937& rng)
	{
		int local_cursor = static_cast<int>(page * offset);
		std::stack<StackFrame> stack;
		StackFrame iframe;
		int max_depth = depth.second;
		size_t index = local_cursor;
		const size_t SAMPLES_SIZE = size.first;
		const size_t FEATURES_SIZE = size.second; 

		iframe.X = X;
		iframe.depth = depth.first;
		iframe.cursor = local_cursor;
		iframe.phase = 0;
		iframe.samples.resize(SAMPLES_SIZE);
		std::iota(iframe.samples.begin(), iframe.samples.end(), 0);
		stack.push(iframe);

		while (!stack.empty())
		{
			StackFrame& frame = stack.top();
			const size_t n_samples = frame.samples.size();
			const size_t n_features = FEATURES_SIZE;
			std::mt19937 rng(std::random_device{} ());

			switch(frame.phase)
			{
			case 0:
			{
				size_t m = static_cast<size_t>(std::sqrt(FEATURES_SIZE));
				std::vector<int> selected_features(FEATURES_SIZE);
				std::iota(selected_features.begin(), selected_features.end(), 0);
				std::shuffle(selected_features.begin(), selected_features.end(), rng);
				selected_features.resize(m);

				std::vector<int> labels;
	            labels.reserve(n_samples);
	            std::transform(
	            	frame.samples.begin(),
	            	frame.samples.end(),
	            	std::back_inserter(labels),
	            	[&] (const int& idx) { return y[idx]; }
	            );

	            if (frame.depth >= max_depth || 
	            	std::all_of(labels.begin(), labels.end(), [&](int l) { 
	            		return l == labels[0]; 
	            	}))
	            {
	            	//this->set_cursor(frame.cursor);
	            	local_cursor = frame.cursor;
	            	//this->add(&DecisionTree::labels, metrics::majority_label(labels));
	            	this->labels[local_cursor] = metrics::majority_label(labels);
	            	index = frame.cursor;
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
					left.reserve(n_samples); right.reserve(n_samples);
					l_labels.reserve(n_samples); r_labels.reserve(n_samples);

					// local variable (omp)
					float split_gain = 0.0f;
					int split_feature = -1;
					float split_threshold = 0.0f;
					std::vector<int> split_left;
					std::vector<int> split_right;

					std::vector<std::pair<float, int>> sorted_samples;
					sorted_samples.reserve(n_samples);

				#ifdef __USE_OMP__
	            	#pragma omp for schedule(dynamic)
				#endif
					for (size_t f = 0; f < selected_features.size(); f++)
		            {
		            	const int feature = selected_features[f];
		            	const float* Xf = X.data() + feature;

		            	// Sort samples per frame
		            	sorted_samples.clear();
		            	std::transform(
			            	frame.samples.begin(),
			            	frame.samples.end(),
			            	std::back_inserter(sorted_samples),
			            	[&] (const int& idx) { return std::make_pair(Xf[idx * FEATURES_SIZE], idx); }
			            );
		            	std::sort(sorted_samples.begin(), sorted_samples.end());

		            	// Reuse vector
		            	left.clear(); right.clear();
		                l_labels.clear(); r_labels.clear();

		                // Initialization: Fill right
		                for (const auto& [_, idx] : sorted_samples)
		                {
		                	right.emplace_back(idx);
		                	r_labels.emplace_back(y[idx]);
		                }

		                std::unordered_map<int, int> l_counts, r_counts;
		                for (const auto& label : r_labels)
		                {
		                	++r_counts[label];
		                }

		                // left: [0...i], right: [i+1...n-1]
		            	for (size_t i = 0; i < n_samples - 1; ++i)
						{
							const auto& [val0, idx0] = sorted_samples[i];
							const auto& [val1, idx1] = sorted_samples[i + 1];

							const int moved_idx = idx0;
							const int moved_label = y[idx0];

							left.emplace_back(moved_idx);
							l_labels.emplace_back(moved_label);

							l_counts[moved_label]++;
							r_counts[moved_label]--;

							if (y[idx0] == y[idx1]) continue;
							if (left.empty() || left.size() >= n_samples) continue;

							float threshold = (val0 + val1) * 0.5f;
							
							const size_t n_left = left.size();
							const size_t n_right = n_samples - n_left;

							float gain = parent_gini
		                        - (static_cast<float>(n_left) / n_samples)  * metrics::gini(l_counts)
		                        - (static_cast<float>(n_right) / n_samples) * metrics::gini(r_counts);

		                #ifdef __USE_OMP__
		                    if (gain > split_gain) 
		                    {
		                        split_gain = gain;
		                        split_feature = feature;
		                        split_threshold = threshold;
		                        split_left = left;
		                        split_right.clear();

		                        std::transform(
								    sorted_samples.begin() + i + 1,
								    sorted_samples.end(),
								    std::back_inserter(split_right),
								    [](const std::pair<float, int>& p) { return p.second; }
								);
		                    }
		                #else
		                    if (gain > frame.split_gain) 
		                    {
		                        frame.split_gain = gain;
		                        frame.split_feature = feature;
		                        frame.split_threshold = threshold;
		                        frame.split_left = left;
		                        frame.split_right.clear();

		                        std::transform(
								    sorted_samples.begin() + i + 1,
								    sorted_samples.end(),
								    std::back_inserter(frame.split_right),
								    [](const std::pair<float, int>& p) { return p.second; }
								);
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

	            if (frame.split_gain == 0 || frame.cursor >= (page + 1) * offset) 
                {
	                //this->set_cursor(frame.cursor);
	                local_cursor = frame.cursor;
	                //this->add(&DecisionTree::labels, metrics::majority_label(labels));
	                this->labels[local_cursor] = metrics::majority_label(labels);
	                index = frame.cursor;
	                stack.pop();
	                continue;
	            }

	            frame.index = frame.cursor;
	            //this->set_cursor(frame.cursor);
	            local_cursor = frame.cursor;
	            //this->add(&DecisionTree::features, frame.split_feature);
	            //this->add(&DecisionTree::thresholds, frame.split_threshold);
	            this->features[local_cursor] = frame.split_feature;
	            this->thresholds[local_cursor] = frame.split_threshold;
	            
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
	            //this->set_cursor(frame.cursor);
	            local_cursor = frame.cursor;
	            //this->add(&DecisionTree::lefts, frame.l_root);
	            this->lefts[local_cursor] = frame.l_root;
	            
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
	            //this->set_cursor(frame.cursor);
	            local_cursor = frame.cursor;
	            //this->add(&DecisionTree::rights, frame.r_root);
	            this->rights[local_cursor] = frame.r_root;
	            
	            index = frame.index;
	            stack.pop();
				break;
			}

			}
		}
	    
	    return index;
	}
}