#include "../cereal/types/polymorphic.hpp"
#include "../cereal/archives/binary.hpp"
#include "DecisionTree.hpp"
#include "FastForest.hpp"
#include "IDecisionNode.hpp"

CEREAL_REGISTER_TYPE(epsilon::ml::rf::structural::DecisionTree)
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    epsilon::ml::rf::structural::IDecisionNode,
    epsilon::ml::rf::structural::DecisionTree
)

CEREAL_REGISTER_TYPE(epsilon::ml::rf::structural::FastForest)
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    epsilon::ml::rf::structural::IDecisionNode,
    epsilon::ml::rf::structural::FastForest
)