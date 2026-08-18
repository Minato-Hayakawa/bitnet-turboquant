# turboquant-cpp

`turboquant-torch`(TurboQuant, ICLR 2026, arXiv:2504.19874 の非公式PyTorch実装)のアルゴリズムを、
PyTorchに依存しない**ヘッダオンリーC++17ライブラリ**として移植したものです。
`bitnet.cpp`(Microsoft公式のBitNet推論エンジン)のようなC++/GGMLベースの環境に、
外部ランタイムを追加せずリンクできることを目的としています。

依存ライブラリなし。標準ライブラリのみで完結します。

## これは何で、何ではないか

**やっていること**
- Stage 1: ランダムHadamard変換 + Lloyd-Max最適スカラー量子化器(b-1 bit/座標)
- Stage 2: 残差の構造化Hadamard JLスケッチ(1bit/座標) + 符号ビット
- ベクトル単位・KVキャッシュのバッチ単位でのquantize/dequantize
- 実測でMSE低下とbit幅の関係を検証済み(`tests/`参照)

**やっていないこと(正直に書きます)**
- **`bitnet.cpp`本体への実配線はまだしていません。** このライブラリはスタンドアロンの量子化コア/ビルディングブロックです。実際にbitnet.cppの推論ホットパスに組み込むには、下記「bitnet.cppへの統合」セクションの作業が別途必要です。
- Stage 2は密Gaussian QJLではなく、固定seedの構造化Hadamard JLスケッチを使用します。これにより量子化・復元をO(d²)からO(d log d)へ下げ、GPU/CPUカーネル化可能な固定手順にしました。残差は1-bit符号から点推定で復元するため、論文の厳密な内積不偏推定そのものではありません。必ず実モデルで精度を検証してください。
- Pythonリファレンス実装にある「sliding window残差バッファ」「outlier channel routing」「GQA対応bit自動調整」「per-layer adaptive bit allocation」は未実装です。同じコア量子化器の上に順次追加できる設計にはなっていますが、今回は含めていません。
- GPU用ggml/CUDAカーネル、およびbitnet.cppのKVストレージへの直結は引き続き実装が必要です。ヘッダ単体はカーネル実装でそのまま再現できる固定seedのHadamard手順になっています。

## ビルド

素の `g++` でも:
```bash
g++ -O3 -std=c++17 -Iinclude tests/test_turboquant.cpp -o test_turboquant
./test_turboquant

g++ -O3 -std=c++17 -Iinclude examples/benchmark.cpp -o benchmark
./benchmark
```

CMakeでも:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
ctest
```
(このサンドボックス環境にcmakeが入っていなかったため、CMake経由のビルドは実行検証していません。素のg++でのコンパイル・実行・テスト全パスは確認済みです。)

## API

```cpp
#include "turboquant/turboquant.hpp"
using namespace turboquant;

// dim=128, bit_width=3 (b-1=2bit for stage1 + 1bit for stage2), unbiased=true
TurboQuant tq(128, 3, /*unbiased=*/true);

std::vector<float> x(128 /* ... fill with data ... */);
QuantizedVector q = tq.quantize(x.data(), x.size());
std::vector<float> x_hat = tq.dequantize(q);

double ratio = tq.compression_ratio(); // fp32比の圧縮率
```

KVキャッシュのバッチ圧縮:
```cpp
#include "turboquant/kv_cache.hpp"
using namespace turboquant;

TurboQuantKVCache cache(/*head_dim=*/128, /*bit_width=*/3, /*unbiased=*/true);

// keys/values: (seq_len x head_dim) row-major
CompressedKV compressed = cache.compress(keys.data(), values.data(), seq_len);

std::vector<float> keys_hat(seq_len * head_dim);
cache.decompress_keys(compressed, keys_hat.data());

auto savings = cache.memory_savings(seq_len); // {original_bytes, compressed_bytes, ratio}
```

## bitnet.cppへの統合(指針・未実装)

`bitnet.cpp` は llama.cpp をベースにしたC++推論エンジンで、KVキャッシュは
`ggml_tensor` として `llama_kv_cache` (llama.cppの `src/llama-kv-cache.cpp` 相当)
が管理しています。統合には難易度が異なる2つのアプローチがあります。

**アプローチA(低リスク・実装容易): オフロード時のみ圧縮**
推論のホットパス自体はfp16/量子化済み重みのまま変更せず、KVキャッシュを
CPUメモリやディスクにオフロードする際にだけこのライブラリで圧縮し、
アテンション計算に戻す直前にdequantizeしてfp16に戻す。
アテンションカーネル自体には手を入れないため、既存コードへの影響が小さく、
「長いコンテキストを保持するためのメモリ削減」という効果だけは得られます。
速度面のメリットはありません(むしろ圧縮/展開のオーバーヘッドが乗ります)。

**アプローチB(高リスク・本命): ホットパスへの直接組み込み**
KVキャッシュのストレージ自体を圧縮形式で持ち、アテンション計算の直前に
ブロック単位でdequantizeするカーネルを書く。効果は大きいですが、
以下のようにllama.cpp内部への変更が必要になります:
- `src/llama-kv-cache.cpp` — KVキャッシュのメモリレイアウト・書き込み/読み出しパス
- `src/llama-graph.cpp` の `build_attn*` 系関数 — アテンションのggml計算グラフ構築
- 新規ggml opの追加、またはCPU/GPUバックエンドへのカスタムカーネル実装
  (このライブラリのquantize/dequantizeロジックをそのカーネルの中身として使う)
- 精度検証: 圧縮あり/なしでperplexityやタスク精度を比較するベンチマークの整備
  (Pythonリファレンスの`downstream task evaluation`と同様の手法)

このリポジトリはBの土台となるコア量子化ロジックを提供するところまでです。
実際にggmlの計算グラフに組み込む作業は、llama.cppのバージョンや
bitnet.cpp側のフォーク差分に強く依存するため、対象バージョンを教えていただければ
次のステップとして具体的なパッチ案を検討できます。

## ディレクトリ構成

```
include/turboquant/
  rng.hpp        乱数生成(xorshift128+, Box-Muller)
  hadamard.hpp   Fast Walsh-Hadamard変換 + ランダム符号反転
  codebook.hpp   Lloyd-Max最適スカラー量子化器
  bitpack.hpp    サブバイト単位のビットパッキング
  turboquant.hpp コアアルゴリズム(Stage1+Stage2)
  kv_cache.hpp   KVキャッシュ用バッチラッパー
tests/test_turboquant.cpp   単体テスト(6ケース、全PASS確認済み)
examples/benchmark.cpp      圧縮率・MSE・速度のベンチマーク
```

## ライセンス

参考にした `turboquant-torch` と同じくMITを想定。実際に組み込む際は
参照元プロジェクトのライセンス表記に従ってください。
