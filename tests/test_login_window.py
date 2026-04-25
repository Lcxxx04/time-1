"""Tests for the login dialog remember-password flow and layout."""

from __future__ import annotations

from types import SimpleNamespace

from PyQt5.QtWidgets import QDialog

from services.auth_service import AuthError
from services.login_preferences import load_remembered_login, save_remembered_login
from ui.login_window import LoginWindow


def _raise_auth_error(store, username, password) -> None:
    raise AuthError("用户名或密码错误。")


def test_login_window_defaults_to_checked_remember_password_and_empty_fields(
    qapp, tmp_path
) -> None:
    window = LoginWindow(store=object(), remembered_login_path=tmp_path / "remembered_login.json")

    assert window._remember_password.isChecked()
    assert window._username.text() == ""
    assert window._password.text() == ""


def test_login_window_prefills_saved_credentials(qapp, tmp_path) -> None:
    path = tmp_path / "remembered_login.json"
    save_remembered_login(path, username="alice", password="secret123")

    window = LoginWindow(store=object(), remembered_login_path=path)

    assert window._username.text() == "alice"
    assert window._password.text() == "secret123"
    assert window._remember_password.isChecked()


def test_successful_login_saves_remembered_credentials(
    qapp, monkeypatch, tmp_path
) -> None:
    path = tmp_path / "remembered_login.json"
    window = LoginWindow(store=object(), remembered_login_path=path)
    window._username.setText("alice")
    window._password.setText("secret123")
    window._remember_password.setChecked(True)

    monkeypatch.setattr(
        "ui.login_window.authenticate_user",
        lambda store, username, password: SimpleNamespace(username=username),
    )

    window._try_login()

    remembered = load_remembered_login(path)
    assert remembered is not None
    assert remembered.username == "alice"
    assert remembered.password == "secret123"
    assert window.result() == QDialog.Accepted


def test_unchecking_remember_password_clears_saved_credentials(qapp, tmp_path) -> None:
    path = tmp_path / "remembered_login.json"
    save_remembered_login(path, username="alice", password="secret123")

    window = LoginWindow(store=object(), remembered_login_path=path)
    window._remember_password.setChecked(False)

    assert load_remembered_login(path) is None
    assert not path.exists()


def test_failed_login_does_not_write_remembered_credentials(
    qapp, monkeypatch, tmp_path
) -> None:
    path = tmp_path / "remembered_login.json"
    window = LoginWindow(store=object(), remembered_login_path=path)
    window._username.setText("alice")
    window._password.setText("wrongpass")
    window._remember_password.setChecked(True)

    monkeypatch.setattr("ui.login_window.authenticate_user", _raise_auth_error)
    monkeypatch.setattr("ui.login_window.QMessageBox.warning", lambda *args, **kwargs: None)

    window._try_login()

    assert load_remembered_login(path) is None
    assert not path.exists()
    assert window.result() == 0


def test_login_window_reserves_space_for_header_text(qapp, tmp_path) -> None:
    window = LoginWindow(store=object(), remembered_login_path=tmp_path / "remembered_login.json")

    assert window.width() >= 660
    assert window.height() >= 360
    assert window._card.maximumWidth() >= 560
    assert window._title_label.minimumHeight() >= 44
    assert window._subtitle_label.minimumHeight() >= 36
    assert window._subtitle_label.wordWrap()
