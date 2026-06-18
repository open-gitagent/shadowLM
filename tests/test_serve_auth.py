"""Studio auth — login/password token minting + bearer validation (no server)."""

import time

from shadowlm.serve import Auth


def test_disabled_when_no_credentials():
    a = Auth(user="admin", password=None, api_key=None)
    assert not a.enabled and a.mode == "none"


def test_password_login_and_token_roundtrip():
    a = Auth(user="admin", password="s3cret", api_key=None)
    assert a.enabled and a.mode == "password"
    assert a.check_login("admin", "s3cret")
    assert not a.check_login("admin", "wrong")
    assert not a.check_login("root", "s3cret")        # wrong user
    token, exp = a.issue_token()
    assert exp > int(time.time())
    assert a.valid_bearer(token)


def test_token_cannot_be_forged_or_tampered():
    a = Auth(user="admin", password="s3cret", api_key=None)
    token, _ = a.issue_token()
    exp_s, _, sig = token.partition(".")
    assert not a.valid_bearer("garbage")
    assert not a.valid_bearer("9999999999.deadbeef")     # bad signature
    assert not a.valid_bearer(f"{int(exp_s) + 3600}.{sig}")  # extended exp, stale sig


def test_expired_token_rejected():
    a = Auth(user="admin", password="s3cret", api_key=None)
    assert not a._valid_token(f"{int(time.time()) - 1}.whatever")


def test_legacy_api_key_mode():
    a = Auth(user="admin", password=None, api_key="KEY123")
    assert a.enabled and a.mode == "apikey"
    assert a.valid_bearer("KEY123")
    assert not a.valid_bearer("nope")
    assert not a.check_login("admin", "KEY123")          # login needs a password


def test_password_takes_precedence_over_apikey_in_mode():
    a = Auth(user="admin", password="pw", api_key="KEY")
    assert a.mode == "password"
    assert a.valid_bearer("KEY")                          # api key still accepted
    assert a.valid_bearer(a.issue_token()[0])             # and password tokens
