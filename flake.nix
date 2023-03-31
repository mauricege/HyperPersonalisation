{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    devenv.url = "github:cachix/devenv";
  };

  outputs = { self, nixpkgs, devenv, ... } @ inputs:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = f: builtins.listToAttrs (map (name: { inherit name; value = f name; }) systems);
    in
    {
      devShells = forAllSystems
        (system:
          let
            pkgs = import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
              };
            };
          in
          {
            default = devenv.lib.mkShell {
              inherit inputs pkgs;
              modules = [
                ({pkgs, config, ... }: {
                  # https://devenv.sh/reference/options/
                  packages = [
                    pkgs.blas.dev
                    pkgs.gcc.out
                    pkgs.glibc
                    pkgs.lapack.dev
                    pkgs.libsndfile.out
                    pkgs.llvm.dev
                    pkgs.stdenv.cc.cc.lib
                    pkgs.zlib
                  ];
                  languages.python = {
                    enable = true;
                    package = pkgs.python310;
                    poetry = {
                      enable = true;
                      package =  pkgs.poetry.override {
                        python3 = pkgs.python310;
                      };
                    };
                  };
                })
              ];
            };
          });
    };
}
