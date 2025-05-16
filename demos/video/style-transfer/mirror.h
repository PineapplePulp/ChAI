
// #ifndef MIRROR_H
// #define MIRROR_H

#ifdef __cplusplus
extern "C" {
#endif

struct cvVideoCapture;
typedef struct cvVideoCapture cvVideoCapture;

int run_mirror(void);

cvVideoCapture get_video_capture(void);

#ifdef __cplusplus
}
#endif

// #endif // MIRROR_H