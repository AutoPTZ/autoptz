from libraries.visca import camera
import time

cam = camera.D100('/dev/cu.usbserial-144210')
cam.init()

cam.zoom_out()
time.sleep(1)
cam.zoom_stop()


"""
{ "addr_set",		"broadcast",	"883001ff",		NO_ARGS },

{ "clear",		"broadcast",	"88010001ff",		NO_ARGS },

{ "power",		"on",		"8101040002ff",		NO_ARGS },
{ "power",		"off",		"8101040003ff",		NO_ARGS },

{ "zoom",		"stop",		"8101040700ff",		NO_ARGS	},
{ "zoom",		"in",		"8101040702ff",		NO_ARGS	},
{ "zoom",		"out",		"8101040703ff",		NO_ARGS	},
{ "zoom",		"in_rate",	"8101040720ff",		ZOOM_SPEED },
{ "zoom",		"out_rate",	"8101040730ff",		ZOOM_SPEED },
{ "zoom",		"direct",	"8101044700000000ff",	ZOOM_OPT_ARG },

{ "focus",		"stop",		"8101040800ff",		NO_ARGS	}, 
{ "focus",		"far",		"8101040802ff",		NO_ARGS	},
{ "focus",		"near",		"8101040803ff",		NO_ARGS	},
{ "focus",		"direct",	"8101044800000000ff",	FOCUS_POS },
{ "focus",		"auto_focus",	"8101043802ff",		NO_ARGS	}, 
{ "focus",		"manual_focus",	"8101043803ff",		NO_ARGS	},

{ "white_bal",		"auto",		"8101043500ff",		NO_ARGS	},
{ "white_bal",		"indoor_mode",	"8101043501ff",		NO_ARGS	},
{ "white_bal",		"outdoor_mode",	"8101043502ff",		NO_ARGS	},
{ "white_bal",		"onepush_mode",	"8101043503ff",		NO_ARGS	},
{ "white_bal",		"onepush_trigr","8101041005ff",		NO_ARGS	},

{ "exposure",		"full_auto",	"8101043900ff",		NO_ARGS	},
{ "exposure",		"manual",	"8101043903ff",		NO_ARGS	},
{ "exposure",		"shutter_priority", "810104390aff",	NO_ARGS	},
{ "exposure",		"iris_priority", "810104390bff",	NO_ARGS	},
{ "exposure",		"bright_mode",	"810104390dff",		NO_ARGS	},

{ "shutter",		"reset",	"8101040a00ff",		NO_ARGS	},
{ "shutter",		"up",		"8101040a02ff",		NO_ARGS	},
{ "shutter",		"down",		"8101040a03ff",		NO_ARGS	}, 
{ "shutter",		"direct",	"8101044a00000000ff",	SHUTR_ARG },

{ "iris",		"reset",	"8101040b00ff",		NO_ARGS	},
{ "iris",		"up",		"8101040b02ff",		NO_ARGS	},
{ "iris",		"down",		"8101040b03ff",		NO_ARGS	}, 
{ "iris",		"direct",	"8101044b00000000ff",	IRIS_ARG },

{ "gain",		"reset",	"8101040c00ff",		NO_ARGS	},
{ "gain",		"up",		"8101040c02ff",		NO_ARGS	},
{ "gain",		"down",		"8101040c03ff",		NO_ARGS	},
{ "gain",		"direct",	"8101044c00000000ff",	GAIN_ARG },

{ "bright",		"reset",	"8101040d00ff",		NO_ARGS	},
{ "bright",		"up",		"8101040d02ff",		NO_ARGS	},
{ "bright",		"down",		"8101040d03ff",		NO_ARGS	},

{ "backlight",		"on",		"8101043302ff",		NO_ARGS	},
{ "backlight",		"off",		"8101043303ff",		NO_ARGS	},

{ "memory",		"reset",	"8101043f0000ff",	MEM_ARG	},
{ "memory",		"set",		"8101043f0100ff",	MEM_ARG	},
{ "memory",		"recall",	"8101043f0200ff",	MEM_ARG	},

{ "datascreen",		"on",		"8101060602ff",		NO_ARGS	},
{ "datascreen",		"off",		"8101060603ff",		NO_ARGS	},

{ "ir_receive",		"on",		"8101060802ff",		NO_ARGS	},
{ "ir_receive",		"off",		"8101060803ff",		NO_ARGS	},

{ "ir_rcvret",		"on",		"81017d01030000ff",	NO_ARGS	},
{ "ir_rcvret",		"off",		"81017d01130000ff",	NO_ARGS	}, 

{ "pantilt",		"up",		"8101060101010301ff",	TILT_SPEED },
{ "pantilt",		"down",		"8101060101010302ff",	TILT_SPEED },
{ "pantilt",		"left",		"8101060101010103ff",	PAN_SPEED },
{ "pantilt",		"right",	"8101060101010203ff",	PAN_SPEED },
{ "pantilt",		"upleft",	"8101060101010101ff",	PT_SPEED },
{ "pantilt",		"upright",	"8101060101010201ff",	PT_SPEED },
{ "pantilt",		"downleft",	"8101060101010102ff",	PT_SPEED },
{ "pantilt",		"downright",	"8101060101010202ff",	PT_SPEED },
{ "pantilt",		"stop",		"8101060101010303ff",	NO_ARGS	}, 
{ "pantilt",		"absolute_pos",	"8101060201010000000000000000ff",PT_POSN },
{ "pantilt",		"relative_pos",	"8101060301010000000000000000ff",PT_POSN },
{ "pantilt",		"home",		"81010604ff",		NO_ARGS	}, 
{ "pantilt",		"reset",	"81010605ff",		NO_ARGS	},

{ "limit",		"set",		"8101060700000000000000000000ff", PT_LMT_SET },
{ "limit",		"clear",	"810106070100070f0f0f070f0f0fff", PT_LMT_CLR },
"""
