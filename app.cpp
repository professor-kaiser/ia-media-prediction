#include "RandomForest/structural/DecisionTree.hpp"
#include "RandomForest/structural/FastForest.hpp"
#include "RandomForest/cereal/archives/binary.hpp"
#include "RandomForest/web/crow_all.h"
#include <algorithm>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdlib>

using epsilon::ml::rf::structural::FastForest;

int main()
{
    crow::App<crow::CORSHandler> app;
    constexpr size_t FEATURES_SIZE = 4;

    app.get_middleware<crow::CORSHandler>()
        .global()
        .headers("Content-Type", "Accept", "Origin")
        .methods("GET"_method, "POST"_method, "OPTIONS"_method)
        .origin("*");

    int port = 18080;
    if (const char* env_p = std::getenv("PORT")) 
    {
        port = std::stoi(env_p);
    }

    std::unique_ptr<FastForest> forest;
    std::ifstream is("model.bin", std::ios::binary);
    cereal::BinaryInputArchive archive(is);
    archive(forest);

    CROW_ROUTE(app, "/rf/prediction/videos")
    .methods("POST"_method) 
    ([&](const crow::request& req, crow::response& res) {
        const auto& body = crow::json::load(req.body);
        crow::json::wvalue result;

        if (!body || !body.has("samples"))
        {
            result["message"] = "Invalid JSON :(";
            res.code = 400;
            res.write(result.dump());
            res.end();

            return;
        }

        const auto& samples = body["samples"];
        std::vector<float> X;
        std::vector<int> y;
        const size_t n_samples = samples.size();

        X.reserve(n_samples);
        y.reserve(n_samples);

        for (const auto& sample : samples)
        {
            std::transform(sample.begin(), sample.end(),
                std::back_inserter(X), 
                [&](const crow::json::rvalue& v) { 
                    return static_cast<float>(v.d()); 
                });
        }

        for (size_t i = 0; i < n_samples; i++)
        {
            y.emplace_back(
                forest->predict(X.data() + i * FEATURES_SIZE, FEATURES_SIZE));
        }

        result["prediction"] = y;
        result["message"] = "Ok ;)";

        res.code = 200;
        res.set_header("Content-Type", "application/json");
        res.write(result.dump());
        res.end();
    });

    CROW_ROUTE(app, "/rf/prediction/video")
    .methods("POST"_method) 
    ([&](const crow::request& req, crow::response& res) {
        const auto& body = crow::json::load(req.body);
        crow::json::wvalue result;

        if (!body || !body.has("sample"))
        {
            result["message"] = "Invalid JSON :(";
            res.code = 400;
            res.write(result.dump());
            res.end();

            return;
        }

        const auto& sample = body["sample"];
        std::vector<float> X;
        X.reserve(sample.size());

        std::transform(sample.begin(), sample.end(),
            std::back_inserter(X), 
            [&](const crow::json::rvalue& v) { 
                return static_cast<float>(v.d()); 
            });

        int y = forest->predict(X);

        result["prediction"] = y;
        result["message"] = "Ok ;)";

        res.code = 200;
        res.set_header("Content-Type", "application/json");
        res.write(result.dump());
        res.end();
    });

    app.port(port)
        .multithreaded()
        .run();
}