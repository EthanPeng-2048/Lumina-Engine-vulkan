// Lumina-Engine Vulkan 1-bit port - standalone correctness + performance harness.
//
// Q1_0 GEMV (token-generation shape, batch = 1). Four kernels are compared:
//   float      : dequantise 1-bit weights to fp32 + fp32 MAC  (stock ggml-vulkan today)
//   int8       : binary weights as int8 signs + per-128 int8 activations + DP4A (ported)
//   rp-float   : same as float, but on the permuted/coalesced layout
//   rp-int8    : the ported kernel on the permuted/coalesced layout
//
// Usage: lumina_q1 <shader_dir> [--rows M] [--cols K] [--iters N] [--device I]

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <random>

#define VK_CHECK(x) do {                                                             \
        VkResult _r = (x);                                                           \
        if (_r != VK_SUCCESS) {                                                      \
            std::fprintf(stderr, "VK error %d at %s:%d\n", (int) _r, __FILE__, __LINE__); \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

#define QK1_0   128
#define Q1_BLK  18          // sizeof(block_q1_0) = 2 (fp16 d) + 16 (bits)
#define LANES   32

static float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) & 0x1u;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1fu;
    const uint32_t m = (uint32_t)(h)       & 0x3ffu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) { f = s << 31; }
        else {
            int ex = -1; uint32_t mm = m;
            do { ex++; mm <<= 1; } while ((mm & 0x400u) == 0);
            mm &= 0x3ffu;
            f = (s << 31) | ((uint32_t)(127 - 15 - ex) << 23) | (mm << 13);
        }
    } else if (e == 31) { f = (s << 31) | 0x7f800000u | (m << 13); }
    else { f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13); }
    float out; std::memcpy(&out, &f, 4); return out;
}
static uint16_t float_to_half(float x) {
    uint32_t f; std::memcpy(&f, &x, 4);
    const uint32_t s = (f >> 16) & 0x8000u;
    int32_t  e = (int32_t)((f >> 23) & 0xffu) - 127 + 15;
    uint32_t m = f & 0x7fffffu;
    if (e <= 0) return (uint16_t) s;
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    return (uint16_t)(s | ((uint32_t) e << 10) | (m >> 13));
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------
static void ref_gemv_float(const std::vector<uint8_t> & A, const std::vector<float> & B,
                           std::vector<float> & D, int M, int K) {
    const int nb = K / QK1_0;
    for (int r = 0; r < M; ++r) {
        double acc = 0.0;
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t * blk = A.data() + (size_t)(r * nb + ib) * Q1_BLK;
            const float dm = half_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
            for (int j = 0; j < QK1_0; ++j) {
                const int bit = (blk[2 + (j >> 3)] >> (j & 7)) & 1;
                const float w = bit ? 1.0f : -1.0f;
                acc += (double) dm * (double) w * (double) B[ib * QK1_0 + j];
            }
        }
        D[r] = (float) acc;
    }
}

static void quantize_q8_128(const std::vector<float> & x, std::vector<int8_t> & q,
                            std::vector<float> & s, int nb) {
    q.resize((size_t) nb * QK1_0);
    s.resize(nb);
    for (int ib = 0; ib < nb; ++ib) {
        float amax = 0.0f;
        for (int j = 0; j < QK1_0; ++j) amax = std::fmax(amax, std::fabs(x[ib * QK1_0 + j]));
        const float d  = amax / 127.0f;
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        s[ib] = d;
        for (int j = 0; j < QK1_0; ++j) {
            int v = (int) std::lround(x[ib * QK1_0 + j] * id);
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[ib * QK1_0 + j] = (int8_t) v;
        }
    }
}

// ---------------------------------------------------------------------------
// Vulkan helpers
// ---------------------------------------------------------------------------
struct Buffer { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkDeviceSize size = 0; };

static uint32_t find_mem_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    std::fprintf(stderr, "no suitable memory type\n"); std::exit(1);
}

static Buffer create_buffer(VkPhysicalDevice phys, VkDevice dev, VkDeviceSize size,
                            VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    Buffer b; b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, b.buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size; ai.memoryTypeIndex = find_mem_type(phys, req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    return b;
}

static void upload(VkPhysicalDevice phys, VkDevice dev, VkCommandPool pool, VkQueue queue,
                   const Buffer & dst, const void * data, VkDeviceSize size) {
    Buffer stg = create_buffer(phys, dev, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void * p = nullptr; VK_CHECK(vkMapMemory(dev, stg.mem, 0, size, 0, &p));
    std::memcpy(p, data, size); vkUnmapMemory(dev, stg.mem);
    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    VkBufferCopy region{}; region.size = size; vkCmdCopyBuffer(cb, stg.buf, dst.buf, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(dev, pool, 1, &cb);
    vkDestroyBuffer(dev, stg.buf, nullptr); vkFreeMemory(dev, stg.mem, nullptr);
}

static std::vector<uint32_t> read_spv(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open shader %s\n", path.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code((size_t) n / 4);
    if (std::fread(code.data(), 1, (size_t) n, f) != (size_t) n) { std::fprintf(stderr, "short read\n"); std::exit(1); }
    std::fclose(f);
    return code;
}

struct Push { uint32_t M, K, nb; };

// ---------------------------------------------------------------------------
// probe the cooperative-matrix (tensor core) capabilities of a device
// ---------------------------------------------------------------------------
static void probe_coopmat(VkInstance instance, VkPhysicalDevice phys) {
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &ext_count, exts.data());
    bool has_khr = false;
    for (auto & e : exts)
        if (!std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) has_khr = true;
    std::printf("  VK_KHR_cooperative_matrix : %s\n", has_khr ? "yes" : "no");
    if (!has_khr) { return; }

    auto get_props = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!get_props) { std::printf("  (entry point not found)\n"); return; }

    uint32_t n = 0;
    get_props(phys, &n, nullptr);
    std::vector<VkCooperativeMatrixPropertiesKHR> props(n);
    for (auto & p : props) { p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR; }
    get_props(phys, &n, props.data());
    std::printf("  cooperative matrix configs: %u\n", n);
    for (auto & p : props) {
        const bool int_ab = (p.AType == VK_COMPONENT_TYPE_SINT8_KHR && p.BType == VK_COMPONENT_TYPE_SINT8_KHR);
        std::printf("    M=%u N=%u K=%u  A/B=%d/%d C=%d%s\n", p.MSize, p.NSize, p.KSize,
                    (int) p.AType, (int) p.BType, (int) p.CType, int_ab ? "   <-- int8xint8" : "");
    }
}

static void fill_q1_weights(std::vector<uint8_t> & A, int M, int K, uint32_t seed) {
    const int nb = K / QK1_0;
    A.assign((size_t) M * nb * Q1_BLK, 0);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> du(0.02f, 0.09f);
    for (int r = 0; r < M; ++r)
        for (int ib = 0; ib < nb; ++ib) {
            uint8_t * blk = A.data() + (size_t)(r * nb + ib) * Q1_BLK;
            const uint16_t d = float_to_half(du(rng));
            blk[0] = (uint8_t)(d & 0xff); blk[1] = (uint8_t)(d >> 8);
            for (int byte = 0; byte < 16; ++byte) blk[2 + byte] = (uint8_t)(rng() & 0xff);
        }
}

// permute the Q1_0 tensor into the block-major, warp-coalesced layout
static void repack_q1(const std::vector<uint8_t> & A, int M, int K,
                      std::vector<uint8_t> & codes, std::vector<uint16_t> & dsc) {
    const int nb    = K / QK1_0;
    const int ng    = (M + LANES - 1) / LANES;
    codes.assign((size_t) ng * nb * LANES * 16, 0);
    dsc.assign((size_t) ng * nb * LANES, 0);
    for (int g = 0; g < ng; ++g)
        for (int ib = 0; ib < nb; ++ib)
            for (int l = 0; l < LANES; ++l) {
                const int r = g * LANES + l;
                const size_t out = ((size_t)(g * nb + ib) * LANES + l);
                if (r < M) {
                    const uint8_t * blk = A.data() + (size_t)(r * nb + ib) * Q1_BLK;
                    std::memcpy(&codes[out * 16], blk + 2, 16);
                    dsc[out] = (uint16_t)(blk[0] | (blk[1] << 8));
                }
            }
}

int main(int argc, char ** argv) {
    std::string shader_dir = (argc > 1 && argv[1][0] != '-') ? argv[1] : ".";
    int M = 4096, K = 4096, iters = 300, device_index = 0;
    bool probe_only = false;
    bool mma_bench = false;
    bool gemm_bench = false;
    bool serial = false;
    bool barrier = false;
    int  mma_wg = 1024, mma_iters = 20000;
    int  kg = 14336;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--rows")   && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--cols")   && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters")  && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device_index = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--probe")) probe_only = true;
        else if (!std::strcmp(argv[i], "--mma-bench")) mma_bench = true;
        else if (!std::strcmp(argv[i], "--gemm")) gemm_bench = true;
        else if (!std::strcmp(argv[i], "--serial")) serial = true;
        else if (!std::strcmp(argv[i], "--barrier")) barrier = true;
        else if (!std::strcmp(argv[i], "--wg")     && i + 1 < argc) mma_wg = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--miters") && i + 1 < argc) mma_iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--k")      && i + 1 < argc) kg = std::atoi(argv[++i]);
    }
    if (K % QK1_0) { std::fprintf(stderr, "cols must be a multiple of 128\n"); return 1; }
    const int nb = K / QK1_0;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "lumina-q1"; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance; VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t ndev = 0; vkEnumeratePhysicalDevices(instance, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(instance, &ndev, devs.data());
    std::printf("Vulkan devices:\n");
    for (uint32_t i = 0; i < ndev; ++i) {
        VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(devs[i], &pr);
        std::printf("  [%u] %s\n", i, pr.deviceName);
    }
    if (device_index < 0 || device_index >= (int) ndev) { std::fprintf(stderr, "bad device index\n"); return 1; }
    VkPhysicalDevice phys = devs[device_index];
    VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(phys, &pr);
    std::printf("using device [%d] = %s\n", device_index, pr.deviceName);
    probe_coopmat(instance, phys);
    if (probe_only) { return 0; }

    uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qf.data());
    int compute_qf = -1;
    for (uint32_t i = 0; i < nqf; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { compute_qf = (int) i; break; }
    if (compute_qf < 0) { std::fprintf(stderr, "no compute queue\n"); return 1; }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = (uint32_t) compute_qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev; VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, (uint32_t) compute_qf, 0, &queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = (uint32_t) compute_qf;
    VkCommandPool pool; VK_CHECK(vkCreateCommandPool(dev, &pci, nullptr, &pool));

    // data
    std::vector<uint8_t> A; std::vector<float> Bf(K);
    fill_q1_weights(A, M, K, 1234u);
    { std::mt19937 rng(99); std::normal_distribution<float> nd(0.0f, 1.0f);
      for (int i = 0; i < K; ++i) Bf[i] = nd(rng); }
    std::vector<int8_t> Bq; std::vector<float> Bs;
    quantize_q8_128(Bf, Bq, Bs, nb);
    // per-128 sum of the int8 activations (used by the FAST kernel's 2*dot(a,bits) - sum(a))
    std::vector<float> Axs(nb);
    for (int ib = 0; ib < nb; ++ib) {
        int32_t s = 0;
        for (int j = 0; j < QK1_0; ++j) s += Bq[(size_t) ib * QK1_0 + j];
        Axs[ib] = (float) s;
    }
    std::vector<uint8_t> codes; std::vector<uint16_t> dsc;
    repack_q1(A, M, K, codes, dsc);

    const VkMemoryPropertyFlags DEV = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const VkBufferUsageFlags SU = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    Buffer bufA    = create_buffer(phys, dev, (VkDeviceSize) A.size(),     SU, DEV);
    Buffer bufCodes= create_buffer(phys, dev, (VkDeviceSize) codes.size(), SU, DEV);
    Buffer bufDsc  = create_buffer(phys, dev, (VkDeviceSize) dsc.size()*2, SU, DEV);
    Buffer bufBf   = create_buffer(phys, dev, (VkDeviceSize) Bf.size()*4,  SU, DEV);
    Buffer bufBq   = create_buffer(phys, dev, (VkDeviceSize) Bq.size(),    SU, DEV);
    Buffer bufBs   = create_buffer(phys, dev, (VkDeviceSize) Bs.size()*4,  SU, DEV);
    Buffer bufAxs  = create_buffer(phys, dev, (VkDeviceSize) Axs.size()*4, SU, DEV);
    Buffer bufD    = create_buffer(phys, dev, (VkDeviceSize) M * 4,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, DEV);
    Buffer bufStg  = create_buffer(phys, dev, (VkDeviceSize) M * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    upload(phys, dev, pool, queue, bufA,     A.data(),     (VkDeviceSize) A.size());
    upload(phys, dev, pool, queue, bufCodes, codes.data(), (VkDeviceSize) codes.size());
    upload(phys, dev, pool, queue, bufDsc,   dsc.data(),   (VkDeviceSize) dsc.size()*2);
    upload(phys, dev, pool, queue, bufBf,    Bf.data(),    (VkDeviceSize) Bf.size()*4);
    upload(phys, dev, pool, queue, bufBq,    Bq.data(),    (VkDeviceSize) Bq.size());
    upload(phys, dev, pool, queue, bufBs,    Bs.data(),    (VkDeviceSize) Bs.size()*4);
    upload(phys, dev, pool, queue, bufAxs,   Axs.data(),   (VkDeviceSize) Axs.size()*4);

    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 96};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 32; dpci.poolSizeCount = 1; dpci.pPoolSizes = &psz;
    VkDescriptorPool dpool; VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));

    struct Kernel { VkPipeline pipe; VkPipelineLayout layout; VkDescriptorSet set; uint32_t groups; };
    auto build_kernel = [&](const char * spv,
                            const std::vector<std::pair<int, Buffer>> & bufs,
                            uint32_t groups, uint32_t push_size = (uint32_t) sizeof(Push)) -> Kernel {
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        auto code = read_spv(shader_dir + "/" + spv);
        smci.codeSize = code.size() * 4; smci.pCode = code.data();
        VkShaderModule mod; VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &mod));

        std::vector<VkDescriptorSetLayoutBinding> binds;
        for (size_t i = 0; i < bufs.size(); ++i)
            binds.push_back({(uint32_t) bufs[i].first, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = (uint32_t) binds.size(); dslci.pBindings = binds.data();
        VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        VkPipelineLayout layout; VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &layout));

        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = mod; cpci.stage.pName = "main";
        cpci.layout = layout;
        VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));
        vkDestroyShaderModule(dev, mod, nullptr);

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dsl;
        VkDescriptorSet set; VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &set));
        std::vector<VkDescriptorBufferInfo> infos;
        for (auto & [b, buf] : bufs) infos.push_back({buf.buf, 0, VK_WHOLE_SIZE});
        std::vector<VkWriteDescriptorSet> writes;
        for (size_t i = 0; i < bufs.size(); ++i) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = set; w.dstBinding = (uint32_t) bufs[i].first; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &infos[i];
            writes.push_back(w);
        }
        vkUpdateDescriptorSets(dev, (uint32_t) writes.size(), writes.data(), 0, nullptr);
        return {pipe, layout, set, groups};
    };

    const uint32_t g_row = (uint32_t)(M + 63) / 64; // one thread per row
    const uint32_t g_sk  = (uint32_t) M;            // one workgroup per row

    Kernel k_float  = build_kernel("q1_gemv_float.comp.spv",    {{0, bufA}, {1, bufBf}, {2, bufD}}, g_row);
    Kernel k_int8   = build_kernel("q1_gemv_int8.comp.spv",     {{0, bufA}, {1, bufBq}, {2, bufBs}, {3, bufD}}, g_row);
    Kernel k_rpf    = build_kernel("q1_gemv_rp_float.comp.spv", {{0, bufCodes}, {1, bufDsc}, {2, bufBf}, {3, bufD}}, g_row);
    Kernel k_rpi    = build_kernel("q1_gemv_rp_int8.comp.spv",  {{0, bufCodes}, {1, bufDsc}, {2, bufBq}, {3, bufBs}, {4, bufD}}, g_row);
    Kernel k_rpiv   = build_kernel("q1_gemv_rp_i8v.comp.spv",   {{0, bufCodes}, {1, bufDsc}, {2, bufBq}, {3, bufBs}, {4, bufD}}, g_row);
    Kernel k_i8vo   = build_kernel("q1_gemv_i8v_orig.comp.spv", {{0, bufA}, {1, bufBq}, {2, bufBs}, {3, bufD}}, g_row);
    Kernel k_skf    = build_kernel("q1_gemv_sk_float.comp.spv", {{0, bufA}, {1, bufBf}, {2, bufD}}, g_sk);
    Kernel k_ski    = build_kernel("q1_gemv_sk_int8.comp.spv",  {{0, bufA}, {1, bufBq}, {2, bufBs}, {3, bufD}}, g_sk);

    // tunable strip-GEMV variants (32 rows per workgroup, K split over NSTRIP warps)
    const uint32_t g_strip = (uint32_t)((M + 31) / 32);
    auto strip_bufs = [&]() { return std::vector<std::pair<int, Buffer>>{
        {0, bufCodes}, {1, bufDsc}, {2, bufBq}, {3, bufBs}, {4, bufD}, {5, bufAxs}}; };
    Kernel t_ns8    = build_kernel("tune_ns8.comp.spv",      strip_bufs(), g_strip);
    Kernel t_ns16   = build_kernel("tune_ns16.comp.spv",     strip_bufs(), g_strip);
    Kernel t_ns8v   = build_kernel("tune_ns8v.comp.spv",     strip_bufs(), g_strip);
    Kernel t_ns8f   = build_kernel("tune_ns8f.comp.spv",     strip_bufs(), g_strip);
    Kernel t_ns8vf  = build_kernel("tune_ns8vf.comp.spv",    strip_bufs(), g_strip);
    Kernel t_ns16vf = build_kernel("tune_ns16vf.comp.spv",   strip_bufs(), g_strip);
    Kernel t_ns16r2 = build_kernel("tune_ns16vf_r2.comp.spv",strip_bufs(), (g_strip + 1) / 2);
    Kernel t_ns8r2  = build_kernel("tune_ns8vf_r2.comp.spv", strip_bufs(), (g_strip + 1) / 2);
    Kernel t_ns8r4  = build_kernel("tune_ns8vf_r4.comp.spv", strip_bufs(), (g_strip + 3) / 4);
    Kernel t_ns4r2  = build_kernel("tune_ns4vf_r2.comp.spv", strip_bufs(), (g_strip + 1) / 2);
    Kernel t_bw     = build_kernel("tune_bw.comp.spv",       strip_bufs(), g_strip);
    Kernel t_bw2    = build_kernel("tune_bw2.comp.spv",      strip_bufs(), g_strip);

    const Push push{(uint32_t) M, (uint32_t) K, (uint32_t) nb};

    auto record = [&](VkCommandBuffer cb, const Kernel & k) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, &k.set, 0, nullptr);
        vkCmdPushConstants(cb, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
        vkCmdDispatch(cb, k.groups, 1, 1);
    };
    auto run_once = [&](const Kernel & k) {
        VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
        VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cb));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &bi)); record(cb, k); VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(queue));
        vkFreeCommandBuffers(dev, pool, 1, &cb);
    };
    auto readback = [&](std::vector<float> & out) {
        VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
        VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cb));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        VkBufferCopy region{}; region.size = (VkDeviceSize) M * 4;
        vkCmdCopyBuffer(cb, bufD.buf, bufStg.buf, 1, &region);
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(queue));
        vkFreeCommandBuffers(dev, pool, 1, &cb);
        void * p = nullptr; VK_CHECK(vkMapMemory(dev, bufStg.mem, 0, (VkDeviceSize) M * 4, 0, &p));
        out.resize(M); std::memcpy(out.data(), p, (size_t) M * 4); vkUnmapMemory(dev, bufStg.mem);
    };
    auto maxerr = [&](const std::vector<float> & ref, const std::vector<float> & cur) {
        double e = 0; for (int i = 0; i < M; ++i) e = std::fmax(e, std::fabs((double) ref[i] - cur[i])); return e;
    };

    if (mma_bench) {
        Buffer outI = create_buffer(phys, dev, (VkDeviceSize) mma_wg * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, DEV);
        Buffer outF = create_buffer(phys, dev, (VkDeviceSize) mma_wg * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, DEV);
        Kernel ci = build_kernel("cm_bench_int8.comp.spv", {{0, outI}}, (uint32_t) mma_wg);
        Kernel cf = build_kernel("cm_bench_f16.comp.spv",  {{0, outF}}, (uint32_t) mma_wg);
        Push mp{(uint32_t) mma_iters, 0u, 0u};
        auto run = [&](Kernel & k, double mac_per_wg_iter, const char * name) {
            auto do_batch = [&](int reps) {
                VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
                VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cb));
                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                VK_CHECK(vkBeginCommandBuffer(cb, &bi));
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, &k.set, 0, nullptr);
                vkCmdPushConstants(cb, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &mp);
                for (int i = 0; i < reps; ++i) vkCmdDispatch(cb, k.groups, 1, 1);
                VK_CHECK(vkEndCommandBuffer(cb));
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
                VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(queue));
                vkFreeCommandBuffers(dev, pool, 1, &cb);
            };
            do_batch(1);
            const int reps = 200;
            auto t0 = std::chrono::high_resolution_clock::now();
            do_batch(reps);
            auto t1 = std::chrono::high_resolution_clock::now();
            const double sec  = std::chrono::duration<double>(t1 - t0).count() / reps;
            const double macs = (double) k.groups * mac_per_wg_iter * (double) mma_iters;
            std::printf("  %-6s %9.1f GMAC/s   %7.1f TFLOPS\n", name, macs / sec / 1e9, 2.0 * macs / sec / 1e12);
        };
        std::printf("\ncooperative-matrix throughput (%d workgroups x %d iters):\n", mma_wg, mma_iters);
        run(cf, 16.0 * 16.0 * 16.0, "f16");
        run(ci, 16.0 * 16.0 * 32.0, "int8");
        return 0;
    }

    if (gemm_bench) {
        struct GemmPush { uint32_t M, N, K, nb; };
        // for --gemm: --rows = M, --cols = N, --k = K

        auto run_gemm = [&](int Mg, int Ncol, int Kg, bool verify, const char * tag) {
            const uint32_t nob = (uint32_t)(Kg / 128);
            std::vector<uint8_t> Ag; fill_q1_weights(Ag, Mg, Kg, 7u);
            std::vector<int8_t> Bqg((size_t) Ncol * Kg);
            std::vector<float>  Bsg((size_t) Ncol * nob);
            std::mt19937 rng(5); std::normal_distribution<float> nd(0.0f, 1.0f);
            for (int c = 0; c < Ncol; ++c) {
                std::vector<float> x(Kg);
                for (int k = 0; k < Kg; ++k) x[k] = nd(rng);
                std::vector<int8_t> q; std::vector<float> s;
                quantize_q8_128(x, q, s, (int) nob);
                for (int k = 0; k < Kg; ++k)  Bqg[(size_t) k * Ncol + c] = q[k];    // [K][N]
                for (uint32_t ib = 0; ib < nob; ++ib) Bsg[(size_t) ib * Ncol + c] = s[ib];
            }
            Buffer bA = create_buffer(phys, dev, (VkDeviceSize) Ag.size(), SU, DEV);
            Buffer bB = create_buffer(phys, dev, (VkDeviceSize) Bqg.size(), SU, DEV);
            Buffer bS = create_buffer(phys, dev, (VkDeviceSize) Bsg.size() * 4, SU, DEV);
            Buffer bD = create_buffer(phys, dev, (VkDeviceSize) Mg * Ncol * 4,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, DEV);
            Buffer stg = create_buffer(phys, dev, (VkDeviceSize) Mg * Ncol * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            upload(phys, dev, pool, queue, bA, Ag.data(), (VkDeviceSize) Ag.size());
            upload(phys, dev, pool, queue, bB, Bqg.data(), (VkDeviceSize) Bqg.size());
            upload(phys, dev, pool, queue, bS, Bsg.data(), (VkDeviceSize) Bsg.size() * 4);

            Kernel gk = build_kernel("q1_gemm_int8_cm.comp.spv",
                                     {{0, bA}, {1, bB}, {2, bS}, {3, bD}},
                                     (uint32_t)((Mg / 16) * (Ncol / 64)), sizeof(GemmPush));
            GemmPush gp{(uint32_t) Mg, (uint32_t) Ncol, (uint32_t) Kg, nob};

            auto dispatch = [&](int reps) {
                VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
                VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cb));
                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                VK_CHECK(vkBeginCommandBuffer(cb, &bi));
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gk.pipe);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gk.layout, 0, 1, &gk.set, 0, nullptr);
                vkCmdPushConstants(cb, gk.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GemmPush), &gp);
                for (int i = 0; i < reps; ++i) vkCmdDispatch(cb, (uint32_t)(Mg / 16), (uint32_t)(Ncol / 64), 1);
                VK_CHECK(vkEndCommandBuffer(cb));
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
                VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(queue));
                vkFreeCommandBuffers(dev, pool, 1, &cb);
            };
            dispatch(1);
            auto t0 = std::chrono::high_resolution_clock::now();
            dispatch(50);
            auto t1 = std::chrono::high_resolution_clock::now();
            const double sec = std::chrono::duration<double>(t1 - t0).count() / 50.0;

            // readback
            {
                VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
                VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cb));
                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                VK_CHECK(vkBeginCommandBuffer(cb, &bi));
                VkBufferCopy region{}; region.size = (VkDeviceSize) Mg * Ncol * 4;
                vkCmdCopyBuffer(cb, bD.buf, stg.buf, 1, &region);
                VK_CHECK(vkEndCommandBuffer(cb));
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
                VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(queue));
                vkFreeCommandBuffers(dev, pool, 1, &cb);
            }
            std::vector<float> Dg((size_t) Mg * Ncol);
            { void * mp = nullptr; VK_CHECK(vkMapMemory(dev, stg.mem, 0, (VkDeviceSize) Mg * Ncol * 4, 0, &mp));
              std::memcpy(Dg.data(), mp, (size_t) Mg * Ncol * 4); vkUnmapMemory(dev, stg.mem); }

            const double flop = 2.0 * Mg * Ncol * Kg;
            std::printf("  %-10s M=%d N=%d K=%d : %8.3f ms  -> %7.1f TFLOPS\n",
                        tag, Mg, Ncol, Kg, sec * 1e3, flop / sec / 1e12);

            if (verify) {
                // CPU reference (per-128 int accumulation, exactly like the shader)
                double maxe = 0, den = 0;
                for (int r = 0; r < Mg; ++r)
                    for (int c = 0; c < Ncol; ++c) {
                        float acc = 0.0f;
                        for (uint32_t ib = 0; ib < nob; ++ib) {
                            const uint8_t * blk = Ag.data() + (size_t)(r * nob + ib) * 18;
                            const float dm = half_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
                            int32_t t32 = 0;
                            for (int j = 0; j < 128; ++j) {
                                const int bit = (blk[2 + (j >> 3)] >> (j & 7)) & 1;
                                const int q = Bqg[(size_t)(ib * 128 + j) * Ncol + c];
                                t32 += bit ? q : -q;
                            }
                            acc += dm * Bsg[(size_t) ib * Ncol + c] * (float) t32;
                        }
                        const float got = Dg[(size_t) r * Ncol + c];
                        maxe = std::fmax(maxe, std::fabs((double) acc - got));
                        den  = std::fmax(den, std::fabs((double) acc));
                    }
                std::printf("               correctness: max abs err = %.3e (rel %.2e)\n", maxe, maxe / den);
            }
        };

        std::printf("\nQ1_0 int8 coopmat GEMM:\n");
        run_gemm(64, 64, 512, true, "verify");
        run_gemm(M, K, kg, false, "perf");
        return 0;
    }

    std::vector<float> Dref(M); ref_gemv_float(A, Bf, Dref, M, K);
    double denom = 0; for (int i = 0; i < M; ++i) denom = std::fmax(denom, std::fabs(Dref[i]));
    std::printf("\ncorrectness (vs fp64 reference, scale=%.3f):\n", denom);
    struct Named { const char * name; Kernel * k; };
    std::vector<Named> kernels = {
        {"float", &k_float}, {"int8", &k_int8},
        {"rp-float", &k_rpf}, {"rp-int8", &k_rpi}, {"rp-i8v", &k_rpiv}, {"i8v-orig", &k_i8vo},
        {"sk-float", &k_skf}, {"sk-int8", &k_ski}};
    std::vector<Named> tuners = {
        {"ns8", &t_ns8}, {"ns16", &t_ns16},
        {"ns8v", &t_ns8v}, {"ns8f", &t_ns8f}, {"ns8vf", &t_ns8vf}, {"ns16vf", &t_ns16vf},
        {"ns16r2", &t_ns16r2}, {"ns8r2", &t_ns8r2}, {"ns8r4", &t_ns8r4},
        {"ns4r2", &t_ns4r2}, {"bw", &t_bw}, {"bw2", &t_bw2}};
    std::vector<float> Dcur;
    for (auto & n : kernels) {
        run_once(*n.k); readback(Dcur);
        const double e = maxerr(Dref, Dcur);
        std::printf("  %-9s max abs err = %.3e   (rel %.2e)\n", n.name, e, e / denom);
    }
    for (auto & n : tuners) {
        if (!std::strcmp(n.name, "bw") || !std::strcmp(n.name, "bw2")) continue;
        run_once(*n.k); readback(Dcur);
        const double e = maxerr(Dref, Dcur);
        std::printf("  %-9s max abs err = %.3e   (rel %.2e)\n", n.name, e, e / denom);
    }

    std::printf("\nperformance (M=%d K=%d, %d iters, %s):\n", M, K, iters,
                serial ? "serial submit" : "batched");
    std::vector<std::pair<std::string, double>> results;
    auto bench = [&](Named & n) {
        run_once(*n.k); // warmup
        const double sec = [&]() {
            if (serial) {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iters; ++i) run_once(*n.k);
                auto t1 = std::chrono::high_resolution_clock::now();
                return std::chrono::duration<double>(t1 - t0).count();
            }
            VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
            VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cb));
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(cb, &bi));
            for (int i = 0; i < iters; ++i) {
                record(cb, *n.k);
                if (barrier) {
                    // force the dispatches to run back to back on the GPU (no CPU round trip),
                    // which is what a real graph with a dependency between the ops looks like
                    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
                }
            }
            VK_CHECK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            auto t0 = std::chrono::high_resolution_clock::now();
            VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(queue));
            auto t1 = std::chrono::high_resolution_clock::now();
            vkFreeCommandBuffers(dev, pool, 1, &cb);
            return std::chrono::duration<double>(t1 - t0).count();
        }();
        const double gmac = (double) M * (double) K * (double) iters / sec / 1e9;
        std::printf("  %-9s %8.3f ms/iter   %7.1f GMAC/s   %6.2f us/dispatch\n",
                    n.name, sec / iters * 1e3, gmac, sec / iters * 1e6);
        results.push_back({n.name, gmac});
    };
    for (auto & n : kernels) bench(n);
    for (auto & n : tuners) bench(n);

    auto gmacs = [&](const char * a) { for (auto & r : results) if (r.first == a) return r.second; return 0.0; };
    {
        double best_float = std::fmax(gmacs("sk-float"), gmacs("rp-float"));
        double best_int8  = std::fmax(gmacs("sk-int8"), gmacs("rp-int8"));
        double best_port  = std::fmax(std::fmax(best_int8, gmacs("rp-i8v")), gmacs("ns8"));
        std::printf("  best float -> best int8 = %.2fx\n", best_int8 / std::fmax(1.0, best_float));
        std::printf("  float(naive) -> best ported = %.2fx\n", best_port / std::fmax(1.0, gmacs("float")));
    }
    const double wbytes = (double) M * (K / 128.0) * 18.0;
    std::printf("  effective weight read: %.2f MB/iter\n", wbytes / 1e6);
    for (auto & r : results) {
        if (r.first == "float" || r.first == "bw" || r.first == "bw2" || r.first == "ns8" || r.first == "ns8vf")
            std::printf("  bw[%s] = %.1f GB/s\n", r.first.c_str(), wbytes * r.second / ((double) M * K));
    }
    return 0;
}
