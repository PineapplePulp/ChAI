
module Env {

    config type defaultEltType = real(32);

    config param releaseChAI = false;
    param developmentAndTesting = !releaseChAI;

    // Minimum needed rank of dynamicTensor for *any* build
    config param minRankNeeded = 4;

    // Maximum needed rank of dynamicTensor for a release build
    config param maxRankNeeded = 10;

    // Maximum rank of dynamicTensor
    config param maxRank = if developmentAndTesting then minRankNeeded else maxRankNeeded;

    private import Bridge;

    inline proc acceleratorAvailable(): bool do
        return Bridge.acceleratorAvailable();

}