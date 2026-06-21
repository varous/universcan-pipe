{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    # ---- Open3D native dependencies ----
    pkgs.libGL
    pkgs.libGLU
    pkgs.glib
    pkgs.stdenv.cc.cc.lib   # libstdc++ / libgomp
    pkgs.zlib
  ];
  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.libGL pkgs.libGLU pkgs.glib pkgs.stdenv.cc.cc.lib pkgs.zlib
    ];
  };
}
