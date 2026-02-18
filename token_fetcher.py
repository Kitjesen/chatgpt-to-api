"""
辅助脚本：获取和验证 ChatGPT token

用法:
  python token_fetcher.py
"""

import json
import time
import base64

import httpx


def decode_jwt_expiry(token: str) -> float:
    try:
        parts = token.split(".")
        if len(parts) >= 2:
            payload = parts[1]
            payload += "=" * (4 - len(payload) % 4)
            data = json.loads(base64.urlsafe_b64decode(payload))
            return data.get("exp", 0)
    except Exception:
        return 0
    return 0


def verify_access_token(token: str) -> bool:
    headers = {
        "authorization": f"Bearer {token}",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    try:
        resp = httpx.get(
            "https://chatgpt.com/backend-api/models",
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("slug", "?") for m in data.get("models", [])]
            print(f"  [OK] Token 有效! 可用模型: {', '.join(models)}")
            return True
        elif resp.status_code == 401:
            print("  [X] Token 无效或已过期")
            return False
        else:
            print(f"  [?] HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [?] 验证失败: {e}")
        return False


def refresh_with_session_token(session_token: str) -> dict:
    """用 session_token 获取新的 access_token"""
    cookies = {"__Secure-next-auth.session-token": session_token}
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "accept": "application/json",
    }
    resp = httpx.get(
        "https://chatgpt.com/api/auth/session",
        cookies=cookies,
        headers=headers,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    access_token = data.get("accessToken")
    if not access_token:
        raise RuntimeError(f"响应中无 accessToken: {json.dumps(data)[:200]}")

    new_session = None
    for cookie in resp.cookies.jar:
        if cookie.name == "__Secure-next-auth.session-token":
            new_session = cookie.value

    return {
        "access_token": access_token,
        "new_session_token": new_session,
        "user": data.get("user", {}),
    }


def save_to_env(key: str, value: str):
    existing = ""
    try:
        with open(".env", "r") as f:
            existing = f.read()
    except FileNotFoundError:
        pass

    if key in existing:
        lines = existing.split("\n")
        new_lines = []
        for line in lines:
            if line.startswith(f"{key}="):
                new_lines.append(f"{key}={value}")
            else:
                new_lines.append(line)
        with open(".env", "w") as f:
            f.write("\n".join(new_lines))
    else:
        with open(".env", "a") as f:
            f.write(f"\n{key}={value}\n")


def main():
    print("=" * 60)
    print("  ChatGPT Token 工具")
    print("=" * 60)
    print()
    print("选择操作:")
    print("  1. 输入 session_token (推荐，支持自动刷新)")
    print("  2. 输入 access_token (临时使用)")
    print("  3. 用已有 session_token 测试刷新")
    print()

    choice = input("选择 (1/2/3): ").strip()

    if choice == "1":
        print()
        print("获取 session_token 的步骤:")
        print("  1. 浏览器登录 https://chatgpt.com")
        print("  2. F12 打开开发者工具")
        print("  3. Application -> Cookies -> chatgpt.com")
        print("  4. 找到 __Secure-next-auth.session-token")
        print("  5. 复制它的值")
        print()

        session_token = input("粘贴 session_token: ").strip()
        if not session_token:
            print("未输入，退出")
            return

        print("\n用 session_token 获取 access_token 中...")
        try:
            result = refresh_with_session_token(session_token)
            access_token = result["access_token"]
            user = result.get("user", {})
            exp = decode_jwt_expiry(access_token)
            remaining = max(0, exp - time.time())

            print(f"\n  [OK] 获取成功!")
            print(f"  用户: {user.get('name', '?')} ({user.get('email', '?')})")
            print(f"  Token 有效期: {remaining/60:.1f} 分钟")
            print(f"  Token 预览: {access_token[:30]}...{access_token[-10:]}")

            save = input("\n保存到 .env? (y/n): ").strip().lower()
            if save == "y":
                save_to_env("CHATGPT_SESSION_TOKEN", session_token)
                save_to_env("CHATGPT_ACCESS_TOKEN", access_token)
                if result.get("new_session_token"):
                    save_to_env("CHATGPT_SESSION_TOKEN", result["new_session_token"])
                print("  [OK] 已保存到 .env")

        except Exception as e:
            print(f"  [X] 失败: {e}")

    elif choice == "2":
        print()
        print("访问 https://chatgpt.com/api/auth/session 复制 accessToken")
        print()
        token = input("粘贴 access_token: ").strip()
        if not token:
            print("未输入，退出")
            return

        print("\n验证中...")
        exp = decode_jwt_expiry(token)
        remaining = max(0, exp - time.time())
        print(f"  有效期剩余: {remaining/60:.1f} 分钟")
        verify_access_token(token)

        save = input("\n保存到 .env? (y/n): ").strip().lower()
        if save == "y":
            save_to_env("CHATGPT_ACCESS_TOKEN", token)
            print("  [OK] 已保存到 .env")

    elif choice == "3":
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("CHATGPT_SESSION_TOKEN="):
                        st = line.strip().split("=", 1)[1]
                        if st:
                            print("\n从 .env 读取 session_token，尝试刷新...")
                            result = refresh_with_session_token(st)
                            access_token = result["access_token"]
                            exp = decode_jwt_expiry(access_token)
                            remaining = max(0, exp - time.time())
                            print(f"  [OK] 刷新成功! 有效期: {remaining/60:.1f} 分钟")
                            verify_access_token(access_token)
                            return
            print("  [X] .env 中未找到 session_token")
        except FileNotFoundError:
            print("  [X] .env 文件不存在")
        except Exception as e:
            print(f"  [X] 失败: {e}")


if __name__ == "__main__":
    main()
