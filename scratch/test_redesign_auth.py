import requests
import re
import sys

for port in [8083, 8000]:
    BASE = f'http://127.0.0.1:{port}'
    session = requests.Session()
    print(f"\n==========================================")
    print(f"Testing WildAI Server on PORT {port}")
    print(f"==========================================")

    # 1. Test Root Landing Page
    r = session.get(f'{BASE}/')
    print('1. Landing page status:', r.status_code)
    assert r.status_code == 200
    assert 'WildAI' in r.text

    # 2. Test Login Page
    r_login_page = session.get(f'{BASE}/login.html')
    print('2. Login page status:', r_login_page.status_code)
    assert r_login_page.status_code == 200
    assert 'Forest Officer Login' in r_login_page.text

    # 3. Test Direct Access to Admin Portal (Both admin-portal.html and admin_portal.html)
    r_admin = session.get(f'{BASE}/admin-portal.html')
    print('3. admin-portal.html status:', r_admin.status_code, 'Length:', len(r_admin.text))
    assert r_admin.status_code == 200
    assert 'Command Center' in r_admin.text or 'Forest Officer' in r_admin.text
    assert 'adminLogoutBtn' in r_admin.text

    # Verify NO emojis in admin-portal.html response
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff\u2600-\u27BF\uD83C-\uDBFF\uDC00-\uDFFF]')
    emojis_in_portal = emoji_pattern.findall(r_admin.text)
    print('   Emojis in admin-portal.html:', len(emojis_in_portal))
    assert len(emojis_in_portal) == 0, f"Found emojis in portal: {emojis_in_portal}"

    # Verify admin_portal.html is also serving the same standard UI
    r_admin_under = session.get(f'{BASE}/admin_portal.html')
    print('   admin_portal.html status:', r_admin_under.status_code, 'Length:', len(r_admin_under.text))
    assert r_admin_under.status_code == 200
    assert 'adminLogoutBtn' in r_admin_under.text
    assert len(emoji_pattern.findall(r_admin_under.text)) == 0

    # 4. Test script.js served has NO emojis
    r_script = session.get(f'{BASE}/script.js')
    print('4. script.js status:', r_script.status_code)
    assert r_script.status_code == 200
    emojis_in_script = emoji_pattern.findall(r_script.text)
    print('   Emojis in script.js:', len(emojis_in_script))
    assert len(emojis_in_script) == 0, f"Found emojis in script.js: {emojis_in_script[:5]}"

    # 5. Test API Login with officer123 / demo123
    r_login = session.post(f'{BASE}/api/login', json={'officerId': 'officer123', 'password': 'demo123'})
    print('5. Login response:', r_login.status_code, r_login.json().get('success'))
    assert r_login.status_code == 200
    assert r_login.json()['success'] is True
    token = r_login.json()['token']
    assert token

    # 6. Test Auth Status when Logged In
    r_auth = session.get(f'{BASE}/api/auth/status')
    print('6. Auth status:', r_auth.json())
    assert r_auth.json()['authenticated'] is True

    # 7. Test Logout API
    r_logout = session.post(f'{BASE}/api/logout')
    print('7. Logout status:', r_logout.status_code, r_logout.json())
    assert r_logout.status_code == 200

    # 8. Test Auth Status After Logout
    r_auth_after = session.get(f'{BASE}/api/auth/status')
    print('8. Auth status after logout:', r_auth_after.json())
    assert r_auth_after.json()['authenticated'] is False

    # 9. Test Functional Backend Endpoints
    print('9. Functional endpoints verification:')
    for ep in ['/api/detections', '/api/alerts', '/api/system/status', '/api/settings']:
        res = session.get(f'{BASE}{ep}')
        print(f'   {ep}: {res.status_code}')
        assert res.status_code == 200

print("\n>>> ALL TESTS ON BOTH 8083 AND 8000 PASSED PERFECTLY! ZERO EMOJIS! <<<")
