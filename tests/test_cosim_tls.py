"""TLS 1.3 co-simulation: server context rules and a Tulpa client end to end."""
import shutil
import ssl
import subprocess

import pytest

mujoco = pytest.importorskip("mujoco")

from physics_engine.engine import PhysicsEngine  # noqa: E402
from tulpa_bridge import toolchain  # noqa: E402
from tulpa_bridge.cosim_server import CoSimServer, CoSimSession, tls_context  # noqa: E402

MODEL = """<mujoco><worldbody><body><joint name="j" type="hinge"/><geom size="0.1" mass="1"/></body></worldbody>
<actuator><motor joint="j"/></actuator></mujoco>"""

needs_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")


def _sh(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    if shutil.which("openssl") is None:
        pytest.skip("needs openssl")
    d = tmp_path_factory.mktemp("pki")
    ec = ["-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes"]
    _sh("openssl", "req", "-x509", *ec, "-keyout", "ca.key", "-out", "ca.pem", "-days", "2", "-subj", "/CN=Test CA", cwd=d)
    _sh("openssl", "req", *ec, "-keyout", "srv.key", "-out", "srv.csr", "-subj", "/CN=localhost", cwd=d)
    (d / "srv.ext").write_text("subjectAltName=DNS:localhost,IP:127.0.0.1\n")
    _sh("openssl", "x509", "-req", "-in", "srv.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
        "-out", "srv.pem", "-days", "2", "-extfile", "srv.ext", cwd=d)
    _sh("openssl", "req", *ec, "-keyout", "cli.key", "-out", "cli.csr", "-subj", "/CN=client", cwd=d)
    (d / "cli.ext").write_text("extendedKeyUsage=clientAuth\n")
    _sh("openssl", "x509", "-req", "-in", "cli.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
        "-out", "cli.pem", "-days", "2", "-extfile", "cli.ext", cwd=d)
    return d


def _server(ctx):
    e = PhysicsEngine()
    e.load_model_from_xml(MODEL)
    return CoSimServer(CoSimSession(e), tls=ctx).start()


def test_non_loopback_requires_tls():
    e = PhysicsEngine()
    e.load_model_from_xml(MODEL)
    with pytest.raises(ValueError, match="needs TLS"):
        CoSimServer(CoSimSession(e), host="0.0.0.0")


@needs_openssl
def test_context_is_tls13_only(pki):
    ctx = tls_context(str(pki / "srv.pem"), str(pki / "srv.key"))
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_3
    srv = _server(ctx)
    try:
        cli = ssl.create_default_context(cafile=str(pki / "ca.pem"))
        cli.maximum_version = ssl.TLSVersion.TLSv1_2
        import socket
        port = srv.server_address[1]
        with pytest.raises(ssl.SSLError):
            with socket.create_connection(("127.0.0.1", port)) as s:
                with cli.wrap_socket(s, server_hostname="localhost") as t:
                    t.sendall(b'{"id":1,"method":"info"}\n')
                    t.recv(10)
    finally:
        srv.stop()


TC = toolchain.find()


@needs_openssl
@pytest.mark.skipif(TC is None, reason="tulpa toolchain not found (set TULPA_BIN)")
def test_tulpa_client_over_mutual_tls(pki, tmp_path):
    srv = _server(tls_context(str(pki / "srv.pem"), str(pki / "srv.key"), client_ca=str(pki / "ca.pem")))
    port = srv.server_address[1]
    prog = tmp_path / "c.rav"
    prog.write_text(
        f'let s = cosim::connect("tls://localhost:{port}", {{"ca": "{pki}/ca.pem", "cert": "{pki}/cli.pem", "key": "{pki}/cli.key"}})\n'
        'print(s.security().version, s.info().nu, s.step([0.5], 4).time > 0)\n'
        "s.close()\n"
    )
    try:
        r = TC.run(str(prog), cwd=str(tmp_path), timeout=60)
    finally:
        srv.stop()
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "TLSv1_3 1 true"
