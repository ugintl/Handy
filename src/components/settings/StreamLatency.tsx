import React from "react";
import { useTranslation } from "react-i18next";
import { Dropdown } from "../ui/Dropdown";
import { SettingContainer } from "../ui/SettingContainer";
import { useSettings } from "../../hooks/useSettings";
import type { StreamLatency } from "@/bindings";

interface StreamLatencyProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const StreamLatencySetting: React.FC<StreamLatencyProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const options = [
      {
        value: "fast",
        label: t("settings.advanced.streamLatency.options.fast"),
      },
      {
        value: "balanced",
        label: t("settings.advanced.streamLatency.options.balanced"),
      },
      {
        value: "accurate",
        label: t("settings.advanced.streamLatency.options.accurate"),
      },
    ];
    const selected = (getSetting("stream_latency") ||
      "balanced") as StreamLatency;

    return (
      <SettingContainer
        title={t("settings.advanced.streamLatency.title")}
        description={t("settings.advanced.streamLatency.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
        tooltipPosition="bottom"
      >
        <Dropdown
          options={options}
          selectedValue={selected}
          onSelect={(value) =>
            updateSetting("stream_latency", value as StreamLatency)
          }
          disabled={isUpdating("stream_latency")}
        />
      </SettingContainer>
    );
  },
);
