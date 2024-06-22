{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "pipzone";
  targetPkgs = pkgs: (with pkgs; [
    stdenv.cc.cc
    zlib
  ]);
  runScript = "fish";
}).env
