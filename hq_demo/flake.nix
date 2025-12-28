{
  description = "Python Shell";

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
          config.allowUnfree = true;
        };
        nvidiaPackage = pkgs.linuxPackages.nvidiaPackages.stable;
      in
      {
        devShell = pkgs.mkShell rec {
          buildInputs = with pkgs; [
            pkgs.python312
            pkgs.uv
            stdenv.cc.cc.lib
            pkgs.zlib
            libGL
            glib
            cudatoolkit
            cudaPackages.cudnn
            nvidiaPackage
            #linuxPackages.nvidia_x11
            #cudaPackages.cuda_cudart
          ];

          env = {
            LD_LIBRARY_PATH = "${
              with pkgs;
              lib.makeLibraryPath [
                zlib
                stdenv.cc.cc.lib
                libGL
                glib
              ]
            }:/run/opengl-driver/lib";
          };
          shellHook = ''
            	    	  export CUDA_PATH=${pkgs.cudatoolkit}
            	              '';
          # echo "uv version: $(uv --version)"
          # echo "python version: $(python --version)"
          # export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
          # export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
          #export CUDA_PATH=${pkgs.cudatoolkit}
          #export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
          #export EXTRA_CCFLAGS="-I/usr/include
          #export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
          #'';
        };
      }
    );
}
