{
  description = "Battlesnake Rules";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: {

    packages.x86_64-linux.battlesnake-rules = nixpkgs.legacyPackages.x86_64-linux.callPackage ./default.nix {};
    packages.x86_64-linux.default = self.packages.x86_64-linux.battlesnake-rules;

  };
}
