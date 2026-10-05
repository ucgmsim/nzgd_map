"""Exercise the July 2026 database through the container and the 2p proxy replica."""

import argparse
import csv
import gzip
import io
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Inspect redirects instead of hiding incorrect upstream hostnames."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ):
        """Return the original redirect response for verification."""
        return


def check_preview(origin: str):
    """Verify prefix routing, data, assets, compression and session redirects."""
    address = urllib.parse.urlsplit(origin)
    if address.scheme != "http" or address.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("Run preview checks against a local HTTP origin only.")
    cookies = CookieJar()
    client = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPCookieProcessor(cookies),
        NoRedirect(),
    )

    def fetch(
        path: str,
        *,
        expected: int = 200,
        form: dict | None = None,
        body: bytes | None = None,
        headers: dict | None = None,
    ):
        data = urllib.parse.urlencode(form).encode() if form is not None else body
        request = urllib.request.Request(
            origin.rstrip("/") + path,
            data=data,
            headers={"Accept-Encoding": "gzip", **(headers or {})},
        )
        try:
            response = client.open(request, timeout=60)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            assert response.status == expected, (path, response.status, expected)
            body = response.read()
            wire_size = len(body)
            if response.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            print(f"PASS {response.status} {path} ({wire_size:,} response bytes)")
            return body, response.headers

    # Docker can return before the new proxy has opened its listening socket.
    for attempt in range(30):
        try:
            fetch("/nzgd/query_help")
            break
        except urllib.error.URLError:
            if attempt == 29:
                raise
            time.sleep(1)

    page, headers = fetch("/nzgd")
    text = page.decode()
    assert "Showing 65801 reports from 58659 NZGD records" in text
    assert headers.get("Content-Encoding") == "gzip", "Full map was not compressed"
    assert "script-src" in headers.get("Content-Security-Policy", "")
    assert 'hx-get="/nzgd/validate"' in text
    page, _ = fetch("/nzgd/?vs30_availability=available")
    assert b"Showing 36342 reports" in page
    page, _ = fetch("/nzgd/?vs30_availability=unavailable")
    assert b"Showing 29459 reports" in page

    javascript = re.search(r'src="(/nzgd/assets/plotly-[^\"]+\.js)"', text)
    assert javascript is not None
    for path in [
        javascript[1],
        "/nzgd/static/htmx.min.js.gz",
        "/nzgd/static/styles.css",
    ]:
        body, _ = fetch(path)
        assert body

    for kind, record_name, record_id in [("cpt", "CPT_3", 3), ("spt", "BH_2307", 2307)]:
        body, _ = fetch(f"/nzgd/{kind}/{record_name}")
        detail = body.decode()
        assert detail.count('class="report-section"') == 2
        download = re.search(
            rf'href="(/nzgd/{kind}/{record_name}/reports/(\d+)/data\.csv)"', detail
        )
        assert download is not None
        body, headers = fetch(download[1])
        assert "attachment" in headers.get("Content-Disposition", "")
        rows = list(csv.DictReader(io.StringIO(body.decode())))
        assert rows
        assert {int(row["nzgd_id"]) for row in rows} == {record_id}
        assert {int(row[f"{kind}_id"]) for row in rows} == {int(download[2])}
        required = {"depth_m", "source_file"}
        required |= (
            {"qc_MPa", "fs_MPa", "u2_MPa"}
            if kind == "cpt"
            else {
                "ISPT_MAIN",
                "ISPT_NVAL",
                "ISPT_REP",
                "number_of_blows",
                "n_value_source",
            }
        )
        assert required <= rows[0].keys()

    body, _ = fetch("/nzgd/validate?query=unknown_field%3E0")
    assert b"Unknown field" in body
    # Browser requests forwarded by reverse proxies can carry several KB of
    # headers. These, plus a submitted query, must fit uWSGI's request buffer.
    proxied = {f"X-Preview-Proxied-Header-{index}": "x" * 1000 for index in range(6)}
    body, _ = fetch("/nzgd/validate?query=deepest_depth%20%3E%203", headers=proxied)
    assert body == b""
    page, _ = fetch(
        "/nzgd/?query=deepest_depth+%3E+3&vs30_availability=all"
        "&vs30_correlation=boore_2004&spt_vs_correlation=brandenberg_2010"
        "&cpt_vs_correlation=andrus_2007_pleistocene&colour_by=vs30"
        "&hist_by=vs30_log_residual",
        headers=proxied,
    )
    assert b"Showing 62807 reports from 56073 NZGD records" in page
    selection = "/nzgd/?query=nzgd_id%3D%3D3&vs30_availability=all"
    # The app's unprivileged user stores uploads in the instance folder; the
    # session reads one back for the overlay, and reset removes it.
    boundary = "nzgd-preview-upload"
    upload = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="return_to"\r\n\r\n'
        f"{selection}\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="geonet_file"; filename="stations.ll"\r\n'
        "Content-Type: text/plain\r\n\r\n"
        "172.6 -43.5 PREVIEW_UPLOADED_STATION\n\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    _, headers = fetch(
        "/nzgd/upload_geonet",
        expected=302,
        body=upload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert headers["Location"] == selection
    body, _ = fetch(selection)
    assert b'"PREVIEW_UPLOADED_STATION"' in body
    _, headers = fetch(
        "/nzgd/clear_geonet", expected=302, form={"return_to": selection}
    )
    assert headers["Location"] == selection
    body, _ = fetch(selection)
    assert b"PREVIEW_UPLOADED_STATION" not in body
    _, headers = fetch(
        "/nzgd/toggle_geonet_visibility", expected=302, form={"return_to": selection}
    )
    assert headers["Location"] == selection
    assert any(
        cookie.name == "nzgd_session" and cookie.path == "/nzgd" for cookie in cookies
    )
    body, _ = fetch(selection)
    assert b"SHOW GEONET STATIONS" in body
    print("Preview checks passed, including the existing 2p path rewrite.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "origin", help="Local preview origin, e.g. http://127.0.0.1:8058"
    )
    check_preview(parser.parse_args().origin)
