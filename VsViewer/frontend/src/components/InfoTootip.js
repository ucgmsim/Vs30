import React, { useState } from "react";

import Tooltip from "@mui/material/Tooltip";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faInfoCircle,
  faCircleExclamation,
} from "@fortawesome/free-solid-svg-icons";

import "assets/tooltip.css";

export default function InfoTooltip({ text, error }) {
  const [open, setOpen] = useState(false);

  const handleTooltipClose = () => setOpen(false);

  const handleTooltipOpen = () => setOpen(true);

  return (
    <Tooltip
      open={open}
      placement="right"
      arrow
      title={<p className="tooltip-text-size">{text}</p>}
    >
      <FontAwesomeIcon
        icon={error ? faCircleExclamation : faInfoCircle}
        size="1x"
        className=""
        onMouseOver={handleTooltipOpen}
        onMouseOut={handleTooltipClose}
        color={error ? "red" : "black"}
      />
    </Tooltip>
  );
}
