package main

import (
	"fmt"
	"os"
	"strings"
)

func printLogTail(path string, maxLines int) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(lines) == 0 || lines[0] == "" {
		return
	}
	start := 0
	if len(lines) > maxLines {
		start = len(lines) - maxLines
	}
	fmt.Fprintln(os.Stderr, "Last log lines:")
	for _, line := range lines[start:] {
		fmt.Fprintln(os.Stderr, "  "+line)
	}
}

func backendExited(exitCh <-chan error) (error, bool) {
	select {
	case err := <-exitCh:
		return err, true
	default:
		return nil, false
	}
}
