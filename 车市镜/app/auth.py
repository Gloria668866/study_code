"""鉴权模块：注册 / 登录 / 当前用户，以及 get_current_user 依赖。

接口（前缀 /api/auth）：
- POST /register {username,password,nickname?}  → 校验重名 + bcrypt 哈希入库
- POST /login    {username,password}            → 校验 → {access_token, user}，更新 last_login_at
- GET  /me       (需 Bearer token)               → 当前用户

get_current_user(Depends)：解析 Authorization: Bearer <token> → user_id → 取用户。
其它受保护接口 Depends(get_current_user) 即可拿到当前 User。
"""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .security import hash_password, verify_password, create_access_token, decode_token

router = APIRouter(prefix="/api/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)  # auto_error=False：自己给中文 401，不抛默认 403


class RegisterIn(BaseModel):
    username: str
    password: str
    nickname: Optional[str] = None


class LoginIn(BaseModel):
    username: str
    password: str


def _user_public(u: User) -> dict:
    """对外返回的用户字段，绝不含 password_hash。"""
    return {
        "id": u.id,
        "username": u.username,
        "nickname": u.nickname,
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
    }


def get_current_user(
    cred: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """从 Bearer token 解析出当前用户；缺失/无效/过期/用户不存在 → 401。"""
    if cred is None or not cred.credentials:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "未提供登录凭证（需 Authorization: Bearer <token>）")
    user_id = decode_token(cred.credentials)
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录凭证无效或已过期")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在")
    return user


@router.post("/register")
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if not body.username.strip() or not body.password:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "用户名和密码不能为空")
    if db.scalar(select(User).where(User.username == body.username)):
        raise HTTPException(status.HTTP_409_CONFLICT, "用户名已存在")
    user = User(
        username=body.username.strip(),
        password_hash=hash_password(body.password),
        nickname=(body.nickname or body.username).strip(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"user": _user_public(user)}


@router.post("/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == body.username))
    # 用户名不存在与密码错误统一报「用户名或密码错误」，不泄露哪个错
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    return {
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
        "user": _user_public(user),
    }


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"user": _user_public(user)}
