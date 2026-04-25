"""Login dialog for local multi-account desktop auth."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.paths import RUNTIME_DIR
from services.auth_service import AuthError, authenticate_user
from services.login_preferences import (
    DEFAULT_REMEMBERED_LOGIN_PATH,
    clear_remembered_login,
    load_remembered_login,
    save_remembered_login,
)
from services.user_store import UserStore, UserStoreError
from ui.password_field import PasswordField
from ui.register_dialog import RegisterDialog


class LoginWindow(QDialog):
    """Authenticate against local user records in ``runtime/users.json``."""

    login_succeeded = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        store: UserStore | None = None,
        remembered_login_path: Path = DEFAULT_REMEMBERED_LOGIN_PATH,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("登录")
        self.setModal(True)
        self.resize(680, 400)
        self.setMinimumSize(660, 390)
        self.setObjectName("loginDialogRoot")

        self._users_path = RUNTIME_DIR / "users.json"
        self._store = store if store is not None else UserStore(self._users_path)
        self._storage_ready = self._ensure_storage_dir()
        self._remembered_login_path = remembered_login_path
        self.current_username: str | None = None

        self._title_label = QLabel("家庭用电负载预测系统", self)
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setObjectName("loginTitle")
        self._title_label.setWordWrap(True)
        self._title_label.setMinimumHeight(48)

        self._subtitle_label = QLabel(
            "数据科技风本地登录入口，凭据仅保存在当前设备。",
            self,
        )
        self._subtitle_label.setAlignment(Qt.AlignCenter)
        self._subtitle_label.setObjectName("pageIntro")
        self._subtitle_label.setWordWrap(True)
        self._subtitle_label.setMinimumHeight(40)

        self._username = QLineEdit(self)
        self._username.setObjectName("loginUsernameInput")
        self._username.setPlaceholderText("请输入用户名")

        self._password_field = PasswordField("请输入密码", self)
        self._password_field.setObjectName("loginPasswordField")
        self._password = self._password_field.line_edit

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)
        form.addRow("用户名", self._username)
        form.addRow("密码", self._password_field)

        self._remember_password = QCheckBox("记住密码", self)
        self._remember_password.setChecked(True)
        self._remember_password.setObjectName("rememberPasswordCheckbox")

        self._login_btn = QPushButton("登录", self)
        self._login_btn.setDefault(True)
        self._login_btn.setObjectName("primaryButton")
        self._login_btn.clicked.connect(self._try_login)
        self._password.returnPressed.connect(self._try_login)

        self._register_btn = QPushButton("注册", self)
        self._register_btn.setObjectName("secondaryButton")
        self._register_btn.clicked.connect(self._on_register_clicked)

        self._login_btn.setEnabled(self._storage_ready)
        self._register_btn.setEnabled(self._storage_ready)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        buttons.addStretch(1)
        buttons.addWidget(self._login_btn)
        buttons.addWidget(self._register_btn)

        card = QFrame(self)
        card.setObjectName("loginCard")
        card.setMinimumWidth(520)
        card.setMaximumWidth(580)
        self._card = card

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 26, 32, 24)
        card_layout.setSpacing(16)
        card_layout.addWidget(self._title_label)
        card_layout.addWidget(self._subtitle_label)
        card_layout.addSpacing(6)
        card_layout.addLayout(form)
        card_layout.addWidget(self._remember_password)
        card_layout.addSpacing(4)
        card_layout.addLayout(buttons)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.addStretch(1)
        root.addWidget(card, alignment=Qt.AlignCenter)
        root.addStretch(1)

        self._remember_password.toggled.connect(self._on_remember_password_toggled)
        self._load_remembered_login()

    def _load_remembered_login(self) -> None:
        remembered = load_remembered_login(self._remembered_login_path)
        if remembered is None:
            return
        self._username.setText(remembered.username)
        self._password.setText(remembered.password)
        self._remember_password.setChecked(remembered.remember_password)

    def _on_register_clicked(self) -> None:
        if not self._storage_ready:
            return
        dialog = RegisterDialog(self._store, self)
        if dialog.exec_() == QDialog.Accepted and dialog.created_username:
            self._username.setText(dialog.created_username)
            self._password.clear()
            self._password.setFocus()

    def _on_remember_password_toggled(self, checked: bool) -> None:
        if checked:
            return
        self._persist_remembered_login()

    def _show_storage_error(self, action: str, _exc: OSError) -> None:
        QMessageBox.critical(
            self,
            "本地账户错误",
            f"无法{action}，请检查本地账户目录是否可用。",
        )

    def _show_store_exception(self, exc: UserStoreError) -> None:
        QMessageBox.critical(self, "本地账户错误", str(exc))

    def _ensure_storage_dir(self) -> bool:
        try:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._show_storage_error("初始化本地账户目录", exc)
            return False
        return True

    def _persist_remembered_login(self) -> None:
        try:
            if self._remember_password.isChecked():
                save_remembered_login(
                    self._remembered_login_path,
                    username=self._username.text(),
                    password=self._password.text(),
                )
            else:
                clear_remembered_login(self._remembered_login_path)
        except OSError as exc:
            action = (
                "保存记住密码设置"
                if self._remember_password.isChecked()
                else "清理记住密码设置"
            )
            self._show_storage_error(action, exc)

    def _try_login(self) -> None:
        if not self._storage_ready:
            return
        try:
            user = authenticate_user(
                self._store, self._username.text(), self._password.text()
            )
        except AuthError as exc:
            message = str(exc)
            if message not in {
                "请输入用户名。",
                "请输入密码。",
                "该账号已禁用。",
                "用户不存在。",
            }:
                message = "用户名或密码错误。"
            QMessageBox.warning(self, "登录失败", message)
            self._password.clear()
            self._password_field.conceal()
            self._password.setFocus()
            return
        except UserStoreError as exc:
            self._show_store_exception(exc)
            return

        self._persist_remembered_login()
        self.current_username = user.username
        self.login_succeeded.emit(user.username)
        self.accept()
