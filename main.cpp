#include "RandomForest/structural/DecisionTree.hpp"
#include "RandomForest/structural/FastForest.hpp"
#include "RandomForest/cereal/archives/binary.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <cstring>
#include <numeric>
#include <atomic>

// g++ -fopenmp -O3 -march=native -std=c++23 main.cpp RandomForest/algorithm/*.cpp RandomForest/structural/*.cpp -D__USE_OMP__ -o main

/*
g++ -I./RandomForest \
-fopenmp -O3 -march=native -std=c++23 \
main.cpp RandomForest/algorithm/*.cpp RandomForest/structural/*.cpp \
-D__USE_OMP__ -o main
*/

using epsilon::ml::rf::structural::DecisionTree;
using epsilon::ml::rf::structural::FastForest;

float normal_dist(std::mt19937& rng, float mean, float stddev)
{
    std::normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}

float lognormal_dist(std::mt19937& rng, float mu_log, float sigma_log)
{
    std::lognormal_distribution<float> dist(mu_log, sigma_log);
    return dist(rng);
}

int main()
{
    const size_t SAMPLES_SIZE = 60000;
    const size_t FEATURES_SIZE = 4;
    std::vector<float> X(SAMPLES_SIZE * FEATURES_SIZE);
    std::vector<float> y(SAMPLES_SIZE);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::vector<int> codes   = {0, 1, 2, 3, 4, 5, 6}; // [240p, 360p, 480p, 720p, 1080p, 1440p, 4K]
    std::vector<double> probs = {0.05, 0.10, 0.15, 0.3, 0.3, 0.07, 0.03};
    std::discrete_distribution<> discrete(probs.begin(), probs.end());

    std::unordered_map<int, std::pair<float, float>> bitrate_params_movies = 
    {
        { 0, {-1.2, 0.25}  }, // 240p  -> ~0.25 Mbps pour ~200 MB
        { 1, {-0.9, 0.28}  }, // 360p  -> ~0.38 Mbps pour ~280 MB
        { 2, {-0.6, 0.31}  }, // 480p  -> ~0.55 Mbps pour ~400 MB
        { 3, {0.0, 0.29}   }, // 720p  -> ~1 Mbps pour ~700 MB
        { 4, {0.6, 0.3}    }, // 1080p -> ~1.8 Mbps pour ~1.3 GB
        { 5, {0.96, 0.36}  }, // 1440p -> ~2.7 Mbps pour ~2 GB
        { 6, {1.61, 0.4}   }  // 2160p (4K) -> ~5 Mbps pour ~3.5 GB
    };

    std::unordered_map<int, std::pair<float, float>> bitrate_params_series = 
    {
        { 0, {-1.1, 0.16}  }, // 240p  -> ~0.28 Mbps pour ~180 MB (45 min)
        { 1, {-0.95, 0.18} }, // 360p  -> ~0.36 Mbps pour ~230 MB (45 min)
        { 2, {-0.79, 0.2}  }, // 480p  -> ~0.45 Mbps pour ~300 MB (45 min)
        { 3, {-0.2, 0.21}  }, // 720p  -> ~0.82 Mbps pour ~550 MB (45 min)
        { 4, {0.4, 0.25}   }, // 1080p -> ~1.5 Mbps pour ~1 GB (45 min)
        { 5, {0.81, 0.23}  }, // 1440p -> ~2.2 Mbps pour ~1.5 GB (45 min)
        { 6, {1.39, 0.3}   }  // 2160p (4K) -> ~4 Mbps pour ~2.7 GB (45 min)
    };

    std::unordered_map<int, std::pair<float, float>> bitrate_params_musics = 
    {
        { 0, {-0.92, 0.11} }, // 240p  -> ~0.33 Mbps pour ~10 MB
        { 1, {-0.72, 0.13} }, // 360p  -> ~0.45 Mbps pour ~14 MB
        { 2, {-0.52, 0.14} }, // 480p  -> ~0.58 Mbps pour ~18 MB
        { 3, {-0.12, 0.16} }, // 720p  -> ~0.91 Mbps pour ~29 MB
        { 4, {0.85, 0.21}  }, // 1080p -> ~2.36 Mbps pour ~75 MB
        { 5, {1.39, 0.22}  }, // 1440p -> ~4.1 Mbps pour ~130 MB
        { 6, {2.39, 0.29}  }  // 2160p (4K) -> ~10.9 Mbps pour ~344 MB
    };

    /*std::unordered_map<int, std::pair<float, float>> bitrate_params_movies = 
    {
        { 0, {-0.6, 0.31} }, // 480p  -> ~0.55 Mbps pour ~400 MB
        { 1, {0.0, 0.29}  }, // 720p  -> ~1 Mbps pour ~700 MB
        { 2, {0.6, 0.3}   }, // 1080p -> ~1.8 Mbps pour ~1.3 GB
        { 3, {0.96, 0.36} }, // 1440p -> ~2.7 Mbps pour ~2 GB
        { 4, {1.61, 0.4}  }  // 2160p (4K) -> ~5 Mbps pour ~3.5 GB
    };
    std::unordered_map<int, std::pair<float, float>> bitrate_params_series = 
    {
        { 0, {-0.79, 0.2} }, // 480p  -> ~0.45 Mbps pour ~300 MB (45 min)
        { 1, {-0.2, 0.21} }, // 720p  -> ~0.82 Mbps pour ~550 MB (45 min)
        { 2, {0.4, 0.25}  }, // 1080p -> ~1.5 Mbps pour ~1 GB (45 min)
        { 3, {0.81, 0.23} }, // 1440p -> ~2.2 Mbps pour ~1.5 GB (45 min)
        { 4, {1.39, 0.3}  }  // 2160p (4K) -> ~4 Mbps pour ~2.7 GB (45 min)
    };
    std::unordered_map<int, std::pair<float, float>> bitrate_params_musics = 
    {
        { 0, {-0.52, 0.14} }, // 480p  -> ~0.58 Mbps pour ~18 MB
        { 1, {-0.12, 0.16} }, // 720p  -> ~0.91 Mbps pour ~29 MB
        { 2, {0.85, 0.21}  }, // 1080p -> ~2.36 Mbps pour ~75 MB
        { 3, {1.39, 0.22}  }, // 1440p -> ~4.1 Mbps pour ~130 MB
        { 4, {2.39, 0.29}  }  // 2160p (4K) -> ~10.9 Mbps pour ~344 MB
    };*/

    auto generate_class = [&] (
        size_t start, size_t end, 
        std::pair<float, float> duration_params,
        std::unordered_map<int, std::pair<float, float>> bitrate_params,
        float label)
    {
        size_t i = start;
        for (; i < end; i++) 
        {
            int code = codes[discrete(rng)];
            auto [mu_duration, sigma_duration] = duration_params;
            auto [mu_bitrate, sigma_bitrate] = bitrate_params[code];
            float duration = lognormal_dist(rng, mu_duration, sigma_duration);
            float bitrate = lognormal_dist(rng, mu_bitrate, sigma_bitrate);

            // 0: Duration
            // 1: Format
            // 2: Bitrate
            // 3: Size
            X[i * FEATURES_SIZE] = duration;
            X[i * FEATURES_SIZE + 1] = static_cast<float>(code);
            X[i * FEATURES_SIZE + 2] = bitrate;
            X[i * FEATURES_SIZE + 3] = duration * 60.f * bitrate / 8.f;
            y[i] = label;
        }
    };

    const size_t CLASS_SIZE = SAMPLES_SIZE / 3;
    generate_class(0,            CLASS_SIZE,   std::make_pair(1.204f, 0.261f), bitrate_params_musics, 0); // Music
    generate_class(CLASS_SIZE,   CLASS_SIZE*2, std::make_pair(3.55f, 0.538f),  bitrate_params_series, 1); // Serie
    generate_class(CLASS_SIZE*2, SAMPLES_SIZE, std::make_pair(4.526f, 0.414f), bitrate_params_movies, 2); // Movie

    int max_depth = static_cast<int>(std::log2(SAMPLES_SIZE));
    std::pair<int, int> depth = {0, max_depth};

    std::cout << "Max depth = " << max_depth << std::endl;

/*
    auto forest = std::make_unique<FastForest>(100);
    forest->build(X, y, {SAMPLES_SIZE, FEATURES_SIZE}, depth, rng);

    std::ofstream os("model.bin", std::ios::binary);
    cereal::BinaryOutputArchive archive(os);
    archive(forest);
*/
    std::unique_ptr<FastForest> forest;
    std::ifstream is("model.bin", std::ios::binary);
    cereal::BinaryInputArchive archive(is);
    archive(forest);

    std::vector<float> test1 = { 3.5f, 1.0f, 0.91f, 38.f   }; // Music
    std::vector<float> test2 = { 39.0f, 1.0f, 0.82f, 210.f }; // Série
    std::vector<float> test3 = { 120.0f, 2.0f, 1.8f, 3500.f }; // Film

    std::cout << "Test 1: Prédiction = " 
              << forest->predict(test1) << " (attendu: 0)" << std::endl;
    std::cout << "Test 2: Prédiction = " 
              << forest->predict(test2) << " (attendu: 1)" << std::endl;
    std::cout << "Test 3: Prédiction = " 
              << forest->predict(test3) << " (attendu: 2)" << std::endl;

    std::atomic<int> correct = 0;
    #pragma omp parallel
    {
        std::vector<float> sample(FEATURES_SIZE);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < SAMPLES_SIZE; ++i) 
        {
            std::memcpy(
                sample.data(),
                X.data() + i * FEATURES_SIZE,
                FEATURES_SIZE * sizeof(float));

            if (forest->predict(sample) == y[i]) 
            {
                correct++;
            }
        }
    }

    std::cout << "Précision sur training set: " 
              << (100.0 * correct.load() / SAMPLES_SIZE) 
              << "%" << std::endl;
    
    return 0;
}