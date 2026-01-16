{
  description = "An environment for the sigmod project";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            llvmPackages.libcxxClang
            llvmPackages.libllvm
            gef
            curl
            git
            cmake
            typst
          ];
          shellHook = ''
            CLANGD_FILE=".clangd"
            CPP_STANDARD="c++20"

            echo "Generating $CLANGD_FILE from \$ clang++ -v output..."

            INCLUDE_PATHS=$(
                ${pkgs.clang}/bin/clang++ -v -E -x c++ - < /dev/null 2>&1 | \
                grep -E '^\s*/nix/store/' | \
                sed 's/^\s*//'
            )

            echo "CompileFlags:" > $CLANGD_FILE
            echo "  Add:" >> $CLANGD_FILE
            echo "    - -std=$CPP_STANDARD" >> $CLANGD_FILE
            OS=$(uname -s); ARCH=$(uname -m)
            if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
                echo "    - -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk" >> $CLANGD_FILE
            fi

            while IFS= read -r PATH_ENTRY; do
                CLEAN_PATH=$(echo "$PATH_ENTRY" | sed -E 's/includ$|include$/include/')
                echo "    - -I$CLEAN_PATH" >> $CLANGD_FILE
            done <<< "$INCLUDE_PATHS"

            echo "    - -O2" >> $CLANGD_FILE

            echo "Generation of $CLANGD_FILE complete."                

            if command -v fish &> /dev/null; then
                exec fish
            fi

          '';
        };
      }
    );
}
