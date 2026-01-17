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
    crow::SimpleApp app;

    int port = 18080;
    if constexpr (const char* env_p = std::getenv("PORT")) 
    {
        port = std::stoi(env_p);
    }

    std::unique_ptr<FastForest> forest;
    std::ifstream is("model.bin", std::ios::binary);
    cereal::BinaryInputArchive archive(is);
    archive(forest);

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