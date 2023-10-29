{ lib, buildGoModule, fetchurl }:

buildGoModule rec {
  pname = "battlesnake-rules";
  version = "v1.2.3";
  patches = [ ./no-browser-patch.diff ];
  src = fetchurl {
    url = "https://github.com/BattlesnakeOfficial/rules/archive/refs/tags/${version}.tar.gz";
    sha256 = "sha256-sxS1faiHK1dht+Pp82Ay3XKRQzY34IVOCMkJyubTHig=";
  };

  vendorSha256 = "sha256-tGOxBhyOPwzguRZY4O2rLoOMaT3EyryjYcO2/2GnVIU=";
}
