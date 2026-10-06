package main

import (
	"net"
	"strings"
)

func displayHost(h string) string {
	if h == "0.0.0.0" {
		return getLocalIP()
	}
	return h
}

func getLocalIP() string {
	addrs, _ := net.InterfaceAddrs()
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() && ipnet.IP.To4() != nil {
			return ipnet.IP.String()
		}
	}
	return "127.0.0.1"
}

func extractFlag(args []string, flag string) (string, []string) {
	var remaining []string
	value := ""
	skip := false
	for i, a := range args {
		if skip {
			skip = false
			continue
		}
		if a == flag && i+1 < len(args) {
			value = args[i+1]
			skip = true
		} else if strings.HasPrefix(a, flag+"=") {
			value = strings.TrimPrefix(a, flag+"=")
		} else {
			remaining = append(remaining, a)
		}
	}
	return value, remaining
}
