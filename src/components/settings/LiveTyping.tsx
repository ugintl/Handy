import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface LiveTypingProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const LiveTyping: React.FC<LiveTypingProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("live_typing") ?? false;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(enabled) => updateSetting("live_typing", enabled)}
        isUpdating={isUpdating("live_typing")}
        label={t("settings.debug.liveTyping.label")}
        description={t("settings.debug.liveTyping.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
