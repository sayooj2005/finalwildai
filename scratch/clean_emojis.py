import re

with open('script.js', 'r', encoding='utf-8') as f:
    text = f.read()

# Emoji pattern
emoji_pattern = re.compile(r'[\U00010000-\U0010ffff\u2600-\u27BF\uD83C-\uDBFF\uDC00-\uDFFF]')

def clean_str(s):
    # Contextual replacements
    s = s.replace('🔴', '')
    s = s.replace('🟡', '')
    s = s.replace('🟢', '')
    s = s.replace('🔋', 'Battery: ')
    s = s.replace('📶', 'Signal: ')
    s = s.replace('✅ Approve', 'Approve')
    s = s.replace('❌ Reject', 'Reject')
    s = s.replace('🔊 Sound Siren', 'Sound Siren')
    s = s.replace('📱 Send SMS', 'Send SMS')
    s = s.replace('✓ Acknowledge', 'Acknowledge')
    s = s.replace('🚨', '')
    s = s.replace('⚠️', '')
    s = s.replace('🐾', '')
    s = s.replace('👤', '')
    s = s.replace('🐅', '')
    s = s.replace('🐆', '')
    s = s.replace('🐘', '')
    s = s.replace('🐻', '')
    s = s.replace('🐗', '')
    s = s.replace('🦌', '')
    s = s.replace('🐒', '')
    s = s.replace('🐕', '')
    s = s.replace('📹', '')
    s = s.replace('🕒', '')
    s = s.replace('🕐', '')
    s = s.replace('🎯', '')
    s = s.replace('📍', '')
    s = s.replace('⚡', '')
    s = s.replace('🧪', '')
    s = s.replace('🛡️', '')
    s = s.replace('🔔', '')
    s = s.replace('🗺️', '')
    s = s.replace('🏥', '')
    s = s.replace('🚔', '')
    s = s.replace('✓', '')
    s = s.replace('❌', '')
    s = s.replace('✅', '')
    # Strip any remaining emojis
    s = emoji_pattern.sub('', s)
    return s

cleaned = clean_str(text)

with open('script.js', 'w', encoding='utf-8') as f:
    f.write(cleaned)

print('Successfully cleaned script.js')
