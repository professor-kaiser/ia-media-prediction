# Étape 1 : Build
FROM ubuntu:22.04 AS build

# Installer les outils
RUN apt-get update && apt-get install -y \
    g++ cmake make libcurl4-openssl-dev libasio-dev libgomp1

WORKDIR /app

# Copier ton code
COPY . .

# Compiler ton projet
RUN g++ -I./RandomForest -fopenmp -O3 -march=native -std=c++20 \
    RandomForest/structural/cereal_registrer.cpp \
    app.cpp RandomForest/algorithm/*.cpp \
    RandomForest/structural/*.cpp \
    -D__USE_OMP__ -o rf_server

# Étape 2 : Runtime léger
FROM ubuntu:22.04

# Installer libgomp pour OpenMP + autres libs
RUN apt-get update && apt-get install -y libgomp1 libasio-dev libcurl4-openssl-dev

WORKDIR /app

# Copier le binaire et le modèle depuis l'étape build
COPY --from=build /app/rf_server .
COPY --from=build /app/model.bin .

EXPOSE 18080

CMD ["./rf_server"]