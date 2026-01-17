#include "metrics.hpp"
#include <unordered_map>
#include <map>
#include <algorithm>

namespace epsilon::ml::rf::algorithm::metrics
{
	int majority_label(const std::vector<int>& indices)
	{
		int n = indices.size();
		std::unordered_map<int, int> freq;
		for (int x : indices)
		{
			++freq[x];
		}

		return std::max_element(freq.begin(), freq.end(),
			[](const auto& a, const auto& b) {
				return a.second < b.second;
			})->first;
	}

	float gini(const std::vector<int>& indices)
	{
		int n = indices.size();
		std::unordered_map<int, int> freq;
		for (int x : indices)
		{
			++freq[x];
		}

		float gini = 1.0f;
		for (const auto& [group, count] : freq)
		{
			float p = static_cast<float>(count) / n;
			gini -= p*p;
		}

		return gini;
	}
	
	std::vector<int> bootstrap(int N, std::mt19937& rng)
	{
		std::uniform_int_distribution<int> dist(0, N-1);
		std::vector<int> indices;
		indices.reserve(N);
		for (int i = 0; i < N; i++)
		{
			indices.emplace_back(dist(rng));
		}

		return indices;
	}
}