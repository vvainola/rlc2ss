#!/usr/bin/env python3
import argparse
import os

def make_identifier(name: str) -> str:
    """Turn filename into a valid C identifier."""
    base = os.path.basename(name)
    base = base.replace('.', '_').replace('-', '_')
    if not base[0].isalpha():
        base = "_" + base
    return base


def main():
    parser = argparse.ArgumentParser(description="Convert binary files to C headers")
    parser.add_argument("--input_files", nargs='+', help="Input binary files")
    parser.add_argument("--output_files", nargs='+', help="Output header files")
    args = parser.parse_args()

    if len(args.input_files) != len(args.output_files):
        parser.error("Number of input files and output files must match.")

    for idx, infile in enumerate(args.input_files):
        outfile = args.output_files[idx]
        varname = make_identifier(os.path.splitext(os.path.basename(outfile))[0]) + "_hexdump"

        with open(infile, "rb") as f:
            data = f.read()

        with open(outfile, "w") as f:
            # Emit C array
            f.write(f"unsigned char {varname}[] = {{\n")

            # Format 12 bytes per line
            for i in range(0, len(data), 12):
                chunk = data[i : i + 12]
                bytestr = ", ".join(f"0x{b:02x}" for b in chunk)
                f.write("  " + bytestr + ",\n")

            f.write("};\n")
            f.write(f"unsigned int {varname}_len = {len(data)};\n")


if __name__ == "__main__":
    main()
