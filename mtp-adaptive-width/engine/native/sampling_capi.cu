// C ABI over crit Inference's token-sampling entry (sampling.h / sampling.cu, retained beside this file),
// so the Python engine calls Core's device sampling law through ctypes on torch's current stream.
#include "sampling.h"
#include <crit/sampling.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ts = crit::techniques::token_sampling;

extern "C" {

std::size_t q38_request_bytes() { return sizeof(ts::Request); }
std::size_t q38_sparse_law_bytes() { return sizeof(ts::SparseLaw); }
std::size_t q38_scratch_floats(std::uint32_t vocab, std::uint32_t rows) { return ts::scratch_floats(vocab, rows); }
std::size_t q38_logprob_scratch_bytes(int vocab, int rows, int n) { return ts::logprob_scratch_bytes(vocab, rows, n); }

// Writes one Request into device memory `request` (synchronously on `stream`).
int q38_set_request(void* request, double temperature, double top_p, double min_p, int top_k, double repetition,
                    double frequency, double presence, std::uint64_t seed, std::uint64_t position, std::int32_t* counts,
                    const std::uint8_t* prompt_seen, cudaStream_t stream) {
    ts::Request host{};
    host.temperature = temperature;
    host.top_p = top_p;
    host.min_p = min_p;
    host.top_k = top_k;
    host.repetition = repetition;
    host.frequency = frequency;
    host.presence = presence;
    host.seed = seed;
    host.position = position;
    host.counts = counts;
    host.prompt_seen = prompt_seen;
    cudaError_t e = cudaMemcpyAsync(request, &host, sizeof(host), cudaMemcpyHostToDevice, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    return static_cast<int>(e);
}

// One draw per (request, column) row with purpose `purpose` (0 sample, 1 draft); staged drafts per column as
// Rows documents; optional sparse laws and logprobs (n alternatives, 0 = off).
int q38_sample(const float* logits, std::uint32_t stride, std::uint32_t vocab, std::uint32_t requests,
               std::uint32_t columns, const void* request, const std::int32_t* staged, std::uint32_t staged_stride,
               std::uint32_t staged_base, std::uint32_t position_base, float* scratch, std::uint32_t purpose,
               std::int32_t* out, void* sparse, std::uint32_t n, float* chosen, std::int32_t* top_ids,
               float* top_logprobs, cudaStream_t stream) {
    ts::Rows rows;
    rows.logits = logits;
    rows.stride = stride;
    rows.vocab = vocab;
    rows.requests = requests;
    rows.columns = columns;
    rows.request = static_cast<const ts::Request*>(request);
    rows.staged = staged;
    rows.staged_stride = staged_stride;
    rows.staged_base = staged_base;
    rows.position_base = position_base;
    rows.scratch = scratch;
    ts::Logprobs lp;
    lp.n = chosen ? n : 0;
    lp.chosen = chosen;
    lp.top_ids = chosen ? top_ids : nullptr;
    lp.top_logprobs = chosen ? top_logprobs : nullptr;
    return static_cast<int>(ts::enqueue_sample(rows, static_cast<crit::core::DrawPurpose>(purpose), out, nullptr,
                                               static_cast<ts::SparseLaw*>(sparse), lp, stream));
}

// Rejection-sampling verify of K drafts per request over K + 1 target columns (greedy drafts when draft_laws is null).
int q38_verify(const float* logits, std::uint32_t stride, std::uint32_t vocab, std::uint32_t requests,
               std::uint32_t columns, const void* request, std::uint32_t position_base, float* scratch,
               const std::int32_t* drafts, const void* draft_laws, std::int32_t* predictions, std::uint32_t n,
               float* chosen, std::int32_t* top_ids, float* top_logprobs, cudaStream_t stream) {
    ts::Verify v;
    v.target.logits = logits;
    v.target.stride = stride;
    v.target.vocab = vocab;
    v.target.requests = requests;
    v.target.columns = columns;
    v.target.request = static_cast<const ts::Request*>(request);
    v.target.position_base = position_base;
    v.target.scratch = scratch;
    v.drafts = drafts;
    v.draft_laws = static_cast<const ts::SparseLaw*>(draft_laws);
    v.predictions = predictions;
    v.logprobs.n = chosen ? n : 0;
    v.logprobs.chosen = chosen;
    v.logprobs.top_ids = chosen ? top_ids : nullptr;
    v.logprobs.top_logprobs = chosen ? top_logprobs : nullptr;
    return static_cast<int>(ts::enqueue_verify(v, stream));
}

int q38_logprobs(const float* logits, int ld, int rows, int vocab, const std::int32_t* chosen, int n,
                 float* chosen_logprob, std::int32_t* top_ids, float* top_logprobs, void* scratch,
                 cudaStream_t stream) {
    return static_cast<int>(ts::enqueue_logprobs(logits, ld, rows, vocab, chosen, n, chosen_logprob, top_ids,
                                                 top_logprobs, scratch, stream));
}

int q38_commit(void* request, std::uint32_t count, std::uint32_t vocab, const std::int32_t* tokens,
               std::uint32_t token_stride, const std::int32_t* emitted, std::uint32_t emitted_stride,
               cudaStream_t stream) {
    return static_cast<int>(ts::enqueue_commit(static_cast<ts::Request*>(request), count, vocab, tokens, token_stride,
                                               emitted, emitted_stride, stream));
}

}  // extern "C"
