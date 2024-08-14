#include <iostream>

#include <ccglib/ccglib.hpp>
#include <ccglib/helper.h>
#include <cudawrappers/cu.hpp>
#include <cxxopts.hpp>

#if defined(HAVE_PMT)
#include <pmt.h>
#endif

cxxopts::Options create_commandline_parser(const char *argv[]) {
  cxxopts::Options options(argv[0], "Packing benchmark");

  options.add_options()("n", "Size(s) of N axis",
                        cxxopts::value<std::vector<size_t>>())(
      "nr_benchmarks", "Number of benchmarks",
      cxxopts::value<size_t>()->default_value(std::to_string(1)))(
      "benchmark_duration", "Approximate benchmark duration (seconds)",
      cxxopts::value<float>()->default_value(std::to_string(4)))(
      "csv", "Format output to CSV",
      cxxopts::value<bool>()->default_value(std::to_string(false)))(
      "direction", "Packing direction (pack or unpack)",
      cxxopts::value<std::string>()->default_value("pack"))(

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

    std::vector<std::string> required_options{"n"};
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
  cxxopts::ParseResult cmdline = parse_commandline(argc, argv);
  std::vector<size_t> N = cmdline["n"].as<std::vector<size_t>>();
  const size_t nr_benchmarks = cmdline["nr_benchmarks"].as<size_t>();
  const float benchmark_duration = cmdline["benchmark_duration"].as<float>();
  const std::string direction = cmdline["direction"].as<std::string>();
  const bool csv = cmdline["csv"].as<bool>();
  const unsigned device_id = cmdline["device"].as<unsigned>();

  // Select packing direction
  std::map<std::string, ccglib::packing::Direction> map_packing_direction{
      {"pack", ccglib::packing::pack}, {"unpack", ccglib::packing::unpack}};
  ccglib::packing::Direction packing_direction =
      map_packing_direction[direction];

  const size_t num_sizes = N.size();

  // Write to cerr to avoid outputing to file if redirect is used
  // add one to number of runs for warmup run
  std::cerr << "Estimated benchmark duration: "
            << nr_benchmarks * benchmark_duration * (num_sizes + 1) << " s"
            << std::endl;

  // GPU init
  cu::init();
  cu::Device device(device_id);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  // Allocate memory for GEMM input / output
  const size_t bytes_out =
      *std::max_element(N.begin(), N.end()) * sizeof(unsigned char);
  const size_t packing_factor =
      sizeof(unsigned) * CHAR_BIT / sizeof(unsigned char);
  const size_t bytes_in = bytes_out * sizeof(unsigned) / packing_factor;
  cu::DeviceMemory d_input(bytes_in);
  cu::DeviceMemory d_output(bytes_out);

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
    std::cout << "#N,runtime_ms,gbytes";
#if defined(HAVE_PMT)
    std::cout << ",watt";
#endif
    std::cout << std::endl;
  }
  for (size_t bench = 0; bench < nr_benchmarks; bench++) {
    for (size_t idx = 0; idx < num_sizes; idx++) {
      ccglib::packing::Packing packing(N[idx], device, stream);

      // Run once to get estimate of runtime per kernel
      cu::Event start;
      cu::Event end;
      stream.record(start);
      packing.Run(d_input, d_output, packing_direction);
      stream.record(end);
      stream.synchronize();
      const size_t nr_iterations =
          ((1000 * benchmark_duration) / end.elapsedTime(start)) + 1;

      // If this is the first iteration, do a warmup run
      if (bench == 0 && idx == 0) {
        for (size_t warmup = 0; warmup < nr_iterations; warmup++) {
          packing.Run(d_input, d_output, packing_direction);
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
        packing.Run(d_input, d_output, packing_direction);
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

      const size_t bytes_out = N[idx] * sizeof(unsigned char);
      const size_t packing_factor =
          sizeof(unsigned) * CHAR_BIT / sizeof(unsigned char);
      const size_t bytes_in = bytes_out * sizeof(unsigned) / packing_factor;

      const double gbytes = (bytes_in + bytes_out) / runtime_ms * 1e-6;

      if (csv) {
        std::cout << N[idx] << "," << runtime_ms << "," << gbytes;
      } else {
        std::cout << std::setw(20) << N[idx] << " " << std::setw(7)
                  << runtime_ms << " ms, " << std::setw(7) << gbytes
                  << " GByte/s";
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
