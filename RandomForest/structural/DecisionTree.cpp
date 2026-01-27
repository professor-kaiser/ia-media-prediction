#include <set>
#include <stack>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <execution>
#include "DecisionTree.hpp"

#ifdef __USE_OMP__
	#include <omp.h>
#endif

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
		boot.reserve(n);

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

	int DecisionTree::predict(float* data, size_t size)
	{
		int node = 0;
        while (lefts[node] != -1 || rights[node] != -1)
        {
            float value = data[features[node]];
            node = value < thresholds[node] 
            	? lefts[node] 
            	: rights[node];
        }

        return labels[node];
	}

	int DecisionTree::build(
	    const std::vector<float>& X,
	    const std::vector<int>& y,
	    const std::pair<size_t, size_t>& size,
	    const std::pair<int, int>& depth,
	    std::mt19937& rng)
	{
	    std::stack<StackFrame> stack;
	    StackFrame iframe;
	    int max_depth = depth.second;
	    size_t index = cursor;
	    const size_t SAMPLES_SIZE = size.second;
	    const size_t FEATURES_SIZE = size.first;
	    size_t n_classes = *std::max_element(y.begin(), y.end()) + 1;

	    std::vector<float> bin_edges((metrics::MAX_BINS + 1) * FEATURES_SIZE);
	    std::vector<uint8_t> X_binned(SAMPLES_SIZE * FEATURES_SIZE);

	    metrics::discretize_t(X_binned, bin_edges, X, size);

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

	        if (frame.phase == 0)
	        {
	            size_t m = static_cast<size_t>(std::sqrt(FEATURES_SIZE));
	            std::vector<int> selected_features(FEATURES_SIZE);
	            std::iota(selected_features.begin(), selected_features.end(), 0);
	            std::shuffle(selected_features.begin(), selected_features.end(), rng);
	            selected_features.resize(m);

	            std::unordered_map<int, int> labels_counts;
	            for (const auto idx : frame.samples)
	            {
	            	++labels_counts[y[idx]];
	            }

	            if (frame.depth >= max_depth || labels_counts.size() == 1)
	            {
	                this->cursor = frame.cursor;
	                this->add(&DecisionTree::labels, metrics::majority_label(labels_counts));
	                index = frame.cursor;
	                stack.pop();
	                continue;
	            }

	            frame.split_gain = 0;
	            frame.split_feature = -1;

	            float parent_gini = metrics::gini(labels_counts);

	        #ifdef __USE_OMP__
	            #pragma omp parallel
	            {
	            	float split_gain = 0.0f;
	                int split_feature = -1;
	                float split_threshold = 0.0f;
	                std::vector<int> split_left, split_right;
	        #endif
	                std::unordered_map<int, int> l_counts, r_counts;
	                std::vector<int> left, binned_indices;

	                left.reserve(n_samples);
	                binned_indices.reserve(n_samples);

	            #ifdef __USE_OMP__
	                #pragma omp for schedule(static)
	            #endif
	                for (size_t f = 0; f < selected_features.size(); f++)
	                {
	                    const int feature = selected_features[f];
	                    const float* edges = bin_edges.data() + feature * (metrics::MAX_BINS + 1);
	                    const uint8_t* Xf_binned = X_binned.data() + feature * SAMPLES_SIZE;
	                    
	                    left.clear();
	                    l_counts.clear();
	                    r_counts = labels_counts;

	      				binned_indices.clear();
	                    binned_indices.assign(frame.samples.begin(), frame.samples.end());
	                    std::sort(binned_indices.begin(), binned_indices.end(),
	                    	[&] (const int i, const int j) {
	                    		return Xf_binned[i] < Xf_binned[j];
	                    	});
	                    
	                    #pragma omp simd
	                    for (size_t i = 0; i < n_samples - 1; ++i)
	                    {
	                    	if (i + metrics::PREFETCH_DISTANCE < n_samples)
	                    		__builtin_prefetch(&binned_indices[i + metrics::PREFETCH_DISTANCE], 0, 1);

	                    	const int idx0 = binned_indices[i];
	                    	const int idx1 = binned_indices[i + 1];
	                    	const uint8_t bin0 = Xf_binned[idx0];
	                    	const uint8_t bin1 = Xf_binned[idx1];
	                        const int moved_label = y[idx0];

	                        left.emplace_back(idx0);
	                        l_counts[moved_label]++;
	                        r_counts[moved_label]--;

	                        if (bin0 == bin1) continue;
	                        if (left.empty() || left.size() >= n_samples) continue;

	                        float threshold = edges[bin1];
	                        
	                        const size_t n_left = i + 1;
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
	                            split_right.assign(
	                            	binned_indices.begin() + n_left, binned_indices.end());
	                        }
	                    #else
	                        if (gain > frame.split_gain) 
	                        {
	                            frame.split_gain = gain;
	                            frame.split_feature = feature;
	                            frame.split_threshold = threshold;
	                            frame.split_left = left;
	                            frame.split_right.clear();

	                            frame.split_right.assign(
	                            	binned_indices.begin() + n_left, binned_indices.end());
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
	                this->cursor = frame.cursor;
	                this->add(&DecisionTree::labels, metrics::majority_label(labels_counts));
	                index = frame.cursor;
	                stack.pop();
	                continue;
	            }

	            frame.index = frame.cursor;
	            this->cursor = frame.cursor;
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
	        }

	        else if (frame.phase == 1)
	        {
	            frame.index = index;
	            this->cursor = frame.cursor;
	            this->add(&DecisionTree::lefts, frame.l_root);
	            
	            frame.r_root = frame.index + 1;
	            frame.phase = 2;
	            
	            StackFrame r_frame;
	            r_frame.samples  = std::move(frame.split_right);
	            r_frame.depth    = frame.depth + 1;
	            r_frame.cursor   = frame.r_root;
	            r_frame.phase    = 0;
	            stack.push(r_frame);
	        }

	        else
	        {
	            frame.index = index;
	            this->cursor = frame.cursor;
	            this->add(&DecisionTree::rights, frame.r_root);
	            
	            index = frame.index;
	            stack.pop();
	        }
	    }
	    
	    return index;
	}

	DecisionTree::~DecisionTree()
	{
	}
}