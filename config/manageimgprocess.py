def image_process_req_manupulate(data):
    user_presets = data['vehicle']['user']['user_presets'][0]
    user_settings = data['vehicle']['user'].get('user_settings', [])

    # Start with preset settings
    preset_settings = user_presets['preset_settings']
    output_json = {
        "user_id": user_presets['user_id'],
        "settings": [
            {
                "setting_id": setting['setting_id'],
                "user_preset_id": setting['user_preset_id'],
                "name": setting['setting']['name'],
                "value": setting['value']
            }
            for setting in preset_settings
        ]
    }

    existing_setting_ids = {s['setting_id'] for s in output_json['settings']}

    # Overwrite with user settings or add new ones
    for user_setting in user_settings:
        updated = False
        for setting in output_json['settings']:
            if setting['setting_id'] == user_setting['setting_id']:
                setting['value'] = user_setting['value']
                updated = True
                break

        # If not found in preset, add it
        if not updated and user_setting['setting_id'] not in existing_setting_ids:
            output_json['settings'].append({
                "setting_id": user_setting['setting_id'],
                "user_preset_id": user_presets['id'],
                "name": user_setting['setting']['name'],
                "value": user_setting['value']
            })

    return output_json