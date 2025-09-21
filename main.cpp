//Feautureset[N] -> M*2 -> K -> 1, N...特徴量の数, MとK...中間層ReLUサイズ
//NNUEはたとえば1024 -> 8より512 -> 16とかのほうがコストを下げられる場合があるので、
//TODO: 特徴量設計

typedef Field = [FIELD_HEIGHT; ]

#include<cstdio>

float ClippedReLU(float inputvec) {
    return min(max(x, 0), 1);
}

struct Game {

}

struct NnueAccumulater {

    float v[2][M];

    float* operator[](perspective) {
        return v[perspective];
    }
}

void refresh_accumulater(
    const LinerLayer& layer, //L_0
    NnueAccumulater& new_acc, //result蓄積
    const std::vector<int>& active_features,
    Game perspective
    ) {
    for (int i = 0; i < M; ++i) {
        new_acc[perspective][i] = layer.bias[i];
    }

    for (int a: active_features) {
        for (int i = 0; i < M; ++i) {
            new_acc[perspective][i] += layer.weight[a][i];
        }
    }

}

void update_accumulator(
    const LinerLayer& layer,
    NnueAccumulater& new_acc,
    const NnueAccumulater& prev_acc,
    const std::vector<int>& removed_feautures,
    const std::vecter<int>& added_feautures,
    Game perspective
) {
    for (int i = 0; i < M; ++i) {
        new_acc[perspective][i] = prev_acc[perspective][i];
    }
    for (int r: removed_feautures) {

    }
}
