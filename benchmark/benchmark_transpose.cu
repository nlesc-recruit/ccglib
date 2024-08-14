#include <iostream>

#include <ccglib/ccglib.hpp>
#include <ccglib/helper.h>
#include <cudawrappers/cu.hpp>
#include <cxxopts.hpp>

#if defined(HAVE_PMT)
#include <pmt.h>
#endif

cxxopts::Options create_commandline_parser(const char *argv[]) {
  cxxopts::Options options(argv[0], "Transpose benchmark");

  options.add_options()(
      "b", "Size of batch axis",
      cxxopts::value<size_t>()->default_value(std::to_string(1)))(
      "m", "Size(s) of M axis", cxxopts::value<std::vector<size_t>>())(
      "n", "Size(s) of N axis", cxxopts::value<std::vector<size_t>>())(
      "pad", "Pad input when not multiple of tile size",
      cxxopts::value<bool>()->default_value(std::to_string(false)))(
      "nr_benchmarks", "Number of benchmarks",
      cxxopts::value<size_t>()->default_value(std::to_string(1)))(
      "benchmark_duration", "Approximate benchmark duration (seconds)",
      cxxopts::value<float>()->default_value(std::to_string(4)))(
      "csv", "Format output to CSV",
      cxxopts::value<bool>()->default_value(std::to_string(false)))(
      "precision", "GEMM precision (float32, float16 or int1)",
      cxxopts::value<std::string>())(
      "variant", "GEMM kernel variant (basic or opt)",
      cxxopts::value<std::string>()->default_value("opt"))(

#ifdef HAVE_PMT
      "measure_power", "Measure power usage",
      cxxopts::value<bool>()->default_value(std::to_string(false)))(
#endif
      "device", "GPU device ID",
      cxxopts::value<unsigned>()->default_value(std::to_string(0)))(
      "h,help", "Print help");

  return options;
}

cxxopts::ParseResult parse_commandline(int argc, const char *argv[]) {
  cxxopts::Options options = create_commandline_parser(argv);

  try {
    cxxopts::ParseResult result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      exit(EXIT_SUCCESS);
    }

    std::vector<std::string> required_options{"m", "n", "precision"};
    for (auto &opt : required_options) {
      if (!result.count(opt)) {
        std::cerr << "Required argument missing: " << opt << std::endl;
        std::cerr << "Run " << argv[0] << " -h for help" << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    return result;
  } catch (const cxxopts::exceptions::exception &err) {
    std::cerr << "Error parsing commandline: " << err.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

inline size_t align(const size_t a, const size_t b) {
  return ccglib::helper::ceildiv(a, b) * b;
}

int main(int argc, const char *argv[]) {
  const size_t complex = 2;

  cxxopts::ParseResult cmdline = parse_commandline(argc, argv);
  const size_t B = cmdline["b"].as<size_t>();
  std::vector<size_t> M_input = cmdline["m"].as<std::vector<size_t>>();
  std::vector<size_t> N_input = cmdline["n"].as<std::vector<size_t>>();
  const bool pad = cmdline["pad"].as<bool>();
  const size_t nr_benchmarks = cmdline["nr_benchmarks"].as<size_t>();
  const float benchmark_duration = cmdline["benchmark_duration"].as<float>();
  const std::string precision = cmdline["precision"].as<std::string>();
  const std::string variant = cmdline["variant"].as<std::string>();
  const bool csv = cmdline["csv"].as<bool>();
  const unsigned device_id = cmdline["device"].as<unsigned>();

  // Select GEMM precision
  std::map<std::string, ccglib::mma::Precision> map_gemm_precision{
      {"float32", ccglib::mma::float32},
      {"float16", ccglib::mma::float16},
      {"int1", ccglib::mma::int1}};
  ccglib::mma::Precision gemm_precision = map_gemm_precision[precision];

  // Select GEMM variant
  std::map<std::string, ccglib::mma::Variant> map_gemm_variant{
      {"basic", ccglib::mma::basic}, {"opt", ccglib::mma::opt}};
  ccglib::mma::Variant gemm_variant = map_gemm_variant[variant];

  // Select size of input / output types
  std::map<std::string, size_t> map_input_bits{
      {"float32", sizeof(float) * CHAR_BIT},
      {"float16", sizeof(half) * CHAR_BIT},
      {"int1", 1}};
  const size_t nr_input_bits = map_input_bits[precision];

  // If one of the input M, N arrays is size one and other is bigger,
  // repeat the single value
  const size_t max_size = std::max({M_input.size(), N_input.size()});
  if (M_input.size() == 1) {
    M_input.resize(max_size);
    for (size_t idx = 1; idx < max_size; idx++) {
      M_input[idx] = M_input[0];
    }
  }
  if (N_input.size() == 1) {
    N_input.resize(max_size);
    for (size_t idx = 1; idx < max_size; idx++) {
      N_input[idx] = N_input[0];
    }
  }

  // M, N should be the same length
  if (!((M_input.size() == N_input.size()))) {
    throw std::runtime_error("m, n, k must be the same length");
  }
  const size_t num_sizes = M_input.size();

  // Write to cerr to avoid outputing to file if redirect is used
  // add one to number of runs for warmup run
  std::cerr << "Estimated benchmark duration: "
            << nr_benchmarks * benchmark_duration * (num_sizes + 1) << " s"
            << std::endl;

  // Verify input matrix shape
  const dim3 tile_sizes =
      ccglib::mma::GEMM::GetDimensions(gemm_precision, gemm_variant);

  std::vector<size_t> M(num_sizes);
  std::vector<size_t> N(num_sizes);

  if (pad) {
    for (size_t i = 0; i < num_sizes; i++) {
      M[i] = align(M_input[i], tile_sizes.x);
      N[i] = align(N_input[i], tile_sizes.z);
    }
  } else {
    M = M_input;
    N = N_input;
    for (size_t i = 0; i < num_sizes; i++) {
      if (M[i] % tile_sizes.x != 0) {
        throw std::runtime_error("all m must be a multiple of " +
                                 std::to_string(tile_sizes.x));
      }
      if (N[i] % tile_sizes.z != 0) {
        throw std::runtime_error("all n must be a multiple of " +
                                 std::to_string(tile_sizes.z));
      }
    }
  }

  // Get max size for A, B, C given the input sizes
  size_t num_elements{0};

  for (size_t i = 0; i < num_sizes; i++) {
    num_elements = std::max(num_elements, M[i] * N[i]);
  }

  // GPU init
  cu::init();
  cu::Device device(device_id);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  // Allocate memory for GEMM input / output
  const size_t bytes = B * complex * num_elements * nr_input_bits / CHAR_BIT;
  cu::DeviceMemory d_input(bytes);
  cu::DeviceMemory d_output(bytes);

  // Fill inputs with non-zero data
  stream.memsetAsync(d_input, static_cast<unsigned char>(0xAA), d_input.size());
  stream.memsetAsync(d_output, static_cast<unsigned char>(0xBB),
                     d_output.size());
  stream.synchronize();

#if defined(HAVE_PMT)
  auto sensor = pmt::Create("nvidia", device_id);
#endif

  // Benchmark, time with either cu events or PMT
  if (csv) {
    // Print output header
    std::cout << "#M,N,runtime_ms,gbytes";
#if defined(HAVE_PMT)
    std::cout << ",watt";
#endif
    std::cout << std::endl;
  }
  for (size_t bench = 0; bench < nr_benchmarks; bench++) {
    for (size_t idx = 0; idx < num_sizes; idx++) {
      ccglib::transpose::Transpose transpose(B, M[idx], N[idx], tile_sizes.x,
                                             tile_sizes.z, nr_input_bits,
                                             device, stream);

      // Run once to get estimate of runtime per kernel
      cu::Event start;
      cu::Event end;
      stream.record(start);
      transpose.Run(d_input, d_output);
      stream.record(end);
      stream.synchronize();
      const size_t nr_iterations =
          ((1000 * benchmark_duration) / end.elapsedTime(start)) + 1;

      // If this is the first iteration, do a warmup run
      if (bench == 0 && idx == 0) {
        for (size_t warmup = 0; warmup < nr_iterations; warmup++) {
          transpose.Run(d_input, d_output);
        }
      }
      stream.synchronize();

      // Run the actual benchmark
#if defined(HAVE_PMT)
      pmt::State pmt_start, pmt_end;
      pmt_start = sensor->Read();
#else
      stream.record(start);
#endif
      for (size_t iter = 0; iter < nr_iterations; iter++) {
        transpose.Run(d_input, d_output);
      }
#if defined(HAVE_PMT)
      stream.synchronize();
      pmt_end = sensor->Read();
      const double runtime_ms =
          1000 * pmt::PMT::seconds(pmt_start, pmt_end) / nr_iterations;
#else
      stream.record(end);
      stream.synchronize();
      const double runtime_ms = end.elapsedTime(start) / nr_iterations;
#endif

      const size_t bytes =
          B * complex * M[idx] * N[idx] * nr_input_bits / CHAR_BIT;
      const double gbytes = (2 * bytes) / runtime_ms * 1e-6;

      if (csv) {
        std::cout << M_input[idx] << "," << N_input[idx] << "," << runtime_ms
                  << "," << gbytes;
      } else {
        const std::string shape = std::to_string(M_input[idx]) + "x" +
                                  std::to_string(N_input[idx]) + "x";
        std::cout << std::setw(20) << shape << " " << std::setw(7) << runtime_ms
                  << " ms, " << std::setw(7) << gbytes << " GByte/s";
      }
#if defined(HAVE_PMT)
      const double watts = pmt::PMT::watts(pmt_start, pmt_end);
      if (csv) {
        std::cout << "," << watts;
      } else {
        std::cout << ", " << std::setw(7) << watts << " W";
      }
#endif
      std::cout << std::endl;
    }
  }
}
