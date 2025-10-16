{
    description = "An environment for the sigmod project";
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
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
                    llvmPackages_21.libcxxClang
                    clang-tools
                    libcxx
                    gdb
                    curl
                    git
                    cmake
                ];
            shellHook = ''
                if command -v fish &> /dev/null; then
                    exec fish
                fi
            '';
            };
        }
    );
}
