#include <iostream>

#include <ccglib/ccglib.hpp>
#include <ccglib/common/helper.h>
#include <cudawrappers/cu.hpp>
#include <cxxopts.hpp>

#include <ccglib/common/precision.h>

#if defined(HAVE_PMT)
#include <pmt.h>
#endif

cxxopts::Options create_commandline_parser(const char *argv[]) {
  cxxopts::Options options(argv[0], "GEMM benchmark");

  options.add_options()("b", "Size of batch axis",
                        cxxopts::value<size_t>()->default_value(
                            "1")) // Default value doesn't need std::to_string
      ("m", "Size(s) of M axis", cxxopts::value<std::vector<size_t>>())(
          "n", "Size(s) of N axis", cxxopts::value<std::vector<size_t>>())(
          "k", "Size(s) of K axis", cxxopts::value<std::vector<size_t>>())(
          "pad", "Pad input when not multiple of tile size",
          cxxopts::value<bool>()->default_value(std::to_string(false)))(
          "nr_benchmarks", "Number of benchmarks",
          cxxopts::value<size_t>()->default_value("1"))(
          "benchmark_duration", "Approximate benchmark duration (seconds)",
          cxxopts::value<float>()->default_value("4"))(
          "csv", "Format output to CSV",
          cxxopts::value<bool>()->default_value(std::to_string(false)))(
          "precision_in",
          "GEMM input precision (float32, float16, float8e4m3, float8e5m2, or "
          "int1)",
          cxxopts::value<std::string>()->default_value("float32"))(
          "precision_out", "GEMM output precision (float32, float16, or int1)",
          cxxopts::value<std::string>()->default_value("float32"))(
          "variant", "GEMM kernel variant (basic or opt)",
          cxxopts::value<std::string>()->default_value("opt"))(
          "complex_axis", "Location of complex axis (planar or interleaved)",
          cxxopts::value<std::string>()->default_value("planar"))
#ifdef HAVE_PMT
          ("measure_power", "Measure power usage",
           cxxopts::value<bool>()->default_value(std::to_string(false)))
#endif
              ("device", "GPU device ID",
               cxxopts::value<unsigned>()->default_value("0"))("h,help",
                                                               "Print help");

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

    std::vector<std::string> required_options{"m", "n", "k", "precision_in"};
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

inline double calc_tops(const size_t batch_size, const size_t M, const size_t N,
                        const size_t K, const float runtime) {
  return 8ULL * 1e-9 * batch_size * M * N * K / runtime;
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
  std::vector<size_t> K_input = cmdline["k"].as<std::vector<size_t>>();
  const bool pad = cmdline["pad"].as<bool>();
  const size_t nr_benchmarks = cmdline["nr_benchmarks"].as<size_t>();
  const float benchmark_duration = cmdline["benchmark_duration"].as<float>();
  const std::string precision_in = cmdline["precision_in"].as<std::string>();
  const std::string precision_out = cmdline["precision_out"].as<std::string>();
  const std::string variant = cmdline["variant"].as<std::string>();
  const std::string complex_axis = cmdline["complex_axis"].as<std::string>();
  const bool csv = cmdline["csv"].as<bool>();
  const unsigned device_id = cmdline["device"].as<unsigned>();

  // Select GEMM precision
  const std::map<const std::string, const ccglib::ValueType> map_gemm_precision{
      {"float32", ccglib::ValueType::float32},
      {"float16", ccglib::ValueType::float16},
      {"float8e4m3", ccglib::ValueType::float8e4m3},
      {"float8e5m2", ccglib::ValueType::float8e5m2},
      {"int32", ccglib::ValueType::int32},
      {"int1", ccglib::ValueType::int1}};

  if (map_gemm_precision.find(precision_in) == map_gemm_precision.end() ||
      map_gemm_precision.find(precision_out) == map_gemm_precision.end()) {
    std::cerr << "Invalid precision provided: input=" << precision_in
              << ", output=" << precision_out << std::endl;
    exit(EXIT_FAILURE);
  }
  const ccglib::Precision gemm_precision(map_gemm_precision.at(precision_in),
                                         map_gemm_precision.at(precision_out));

  // Select GEMM variant
  const std::map<const std::string, ccglib::mma::Variant> map_gemm_variant{
      {"basic", ccglib::mma::basic}, {"opt", ccglib::mma::opt}};
  const ccglib::mma::Variant &gemm_variant = map_gemm_variant.at(variant);

  // Select complex axis location
  const std::map<const std::string, const ccglib::ComplexAxisLocation>
      map_gemm_complex_axis{{"planar", ccglib::complex_planar},
                            {"interleaved", ccglib::complex_interleaved}};
  const ccglib::ComplexAxisLocation &gemm_complex_axis_location =
      map_gemm_complex_axis.at(complex_axis);

  // If one of the input M, N, K arrays is size one and others are bigger,
  // repeat the single value
  const size_t max_size =
      std::max({M_input.size(), N_input.size(), K_input.size()});
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
  if (K_input.size() == 1) {
    K_input.resize(max_size);
    for (size_t idx = 1; idx < max_size; idx++) {
      K_input[idx] = K_input[0];
    }
  }

  // M, N, K should be the same length
  if (!((M_input.size() == N_input.size()) &&
        (N_input.size() == K_input.size()))) {
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
  std::vector<size_t> K(num_sizes);
  if (pad) {
    for (size_t i = 0; i < num_sizes; i++) {
      M[i] = align(M_input[i], tile_sizes.x);
      N[i] = align(N_input[i], tile_sizes.y);
      K[i] = align(K_input[i], tile_sizes.z);
    }
  } else {
    M = M_input;
    N = N_input;
    K = K_input;
    // Int1 kernel does not support non-multiples
    if (gemm_precision.input_type == ccglib::ValueType::int1) {
      for (size_t i = 0; i < num_sizes; i++) {
        if (M[i] % tile_sizes.x != 0) {
          throw std::runtime_error("all m must be a multiple of " +
                                   std::to_string(tile_sizes.x));
        }
        if (N[i] % tile_sizes.y != 0) {
          throw std::runtime_error("all n must be a multiple of " +
                                   std::to_string(tile_sizes.y));
        }
        if (K[i] % tile_sizes.z != 0) {
          throw std::runtime_error("all k must be a multiple of " +
                                   std::to_string(tile_sizes.z));
        }
      }
    }
  }

  // Get max size for A, B, C given the input sizes
  size_t num_a{0};
  size_t num_b{0};
  size_t num_c{0};

  for (size_t i = 0; i < num_sizes; i++) {
    num_a = std::max(num_a, M[i] * K[i]);
    num_b = std::max(num_b, N[i] * K[i]);
    num_c = std::max(num_c, M[i] * N[i]);
  }

  // GPU init
  cu::init();
  cu::Device device(device_id);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  const size_t nr_input_bits = gemm_precision.GetInputBits();
  const size_t nr_output_bits = gemm_precision.GetOutputBits();

  // Allocate memory for GEMM input / output
  cu::DeviceMemory d_a(B * complex * num_a * nr_input_bits / CHAR_BIT);
  cu::DeviceMemory d_b(B * complex * num_b * nr_input_bits / CHAR_BIT);
  cu::DeviceMemory d_c(B * complex * num_c * nr_output_bits / CHAR_BIT);

  // Fill inputs with non-zero data
  stream.memsetAsync(d_a, static_cast<unsigned char>(0xAA), d_a.size());
  stream.memsetAsync(d_b, static_cast<unsigned char>(0xBB), d_b.size());
  stream.synchronize();

#if defined(HAVE_PMT)
#if defined(__HIP__)
  auto sensor = pmt::Create("rocm", std::to_string(device_id));
#else
  auto sensor = pmt::Create("nvidia", std::to_string(device_id));
#endif
#endif

  // Benchmark, time with either cu events or PMT
  if (csv) {
    // Print output header
    std::cout << "#M,N,K,runtime_ms,tops";
#if defined(HAVE_PMT)
    std::cout << ",watt,tops_per_joule";
#endif
    std::cout << std::endl;
  }
  for (size_t bench = 0; bench < nr_benchmarks; bench++) {
    for (size_t idx = 0; idx < num_sizes; idx++) {
      ccglib::mma::GEMM gemm(B, M[idx], N[idx], K[idx], device, stream,
                             gemm_precision, gemm_variant,
                             gemm_complex_axis_location);

      // Run once to get estimate of runtime per kernel
      cu::Event start;
      cu::Event end;
      stream.record(start);
      gemm.Run(d_a, d_b, d_c);
      stream.record(end);
      stream.synchronize();
      const size_t nr_iterations =
          ((1000 * benchmark_duration) / end.elapsedTime(start)) + 1;

      // If this is the first iteration, do a warmup run
      if (bench == 0 && idx == 0) {
        for (size_t warmup = 0; warmup < nr_iterations; warmup++) {
          gemm.Run(d_a, d_b, d_c);
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
        gemm.Run(d_a, d_b, d_c);
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

      const double tops =
          calc_tops(B, M_input[idx], N_input[idx], K_input[idx], runtime_ms);

      if (csv) {
        std::cout << M_input[idx] << "," << N_input[idx] << "," << K_input[idx]
                  << "," << runtime_ms << "," << tops;
      } else {
        const std::string shape = std::to_string(M_input[idx]) + "x" +
                                  std::to_string(N_input[idx]) + "x" +
                                  std::to_string(K_input[idx]);
        std::cout << std::setw(20) << shape << " " << std::setw(7) << runtime_ms
                  << " ms, " << std::setw(7) << tops << " TOps/s";
      }
#if defined(HAVE_PMT)
      const double watts = pmt::PMT::watts(pmt_start, pmt_end);
      if (csv) {
        std::cout << "," << watts << "," << tops / watts;
      } else {
        std::cout << ", " << std::setw(7) << watts << " W, " << std::setw(7)
                  << tops / watts << " TOps/J";
      }
#endif
      std::cout << std::endl;
    }
  }
}
